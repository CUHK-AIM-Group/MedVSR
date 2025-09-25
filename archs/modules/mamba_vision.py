#!/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.layers.mlp import SwiGLU, GatedMlp, GluMlp
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.modules.mamba_simple import Mamba

from einops import rearrange, repeat
from pathlib import Path

try:
    from causal_conv1d import causal_conv1d_fn # 1.1.1
except ImportError:
    causal_conv1d_fn = None



def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=16,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        # others
        local_window=False,
        window_size=7,
        use_pe=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        # d_inner 64 d_state 16 d_model 64 ngroups 1 headdim 16
        # print("d_inner", self.d_inner, "d_state", self.d_state, "d_model", self.d_model, "ngroups", self.ngroups, "headdim", self.headdim)
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        # self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, elementwise_affine=True, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.local_window = local_window
        self.use_pe = use_pe
        # print("local_window", self.local_window, "use_pe", self.use_pe)
        if self.local_window:
            self.window_size = window_size

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]

        # 1D Convolution
        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
            )  # (B, L, self.d_inner + 2 * ngroups * d_state)
            xBC = xBC[:, :seqlen, :]
        else:
            xBC = causal_conv1d_fn(
                x=xBC.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            ).transpose(1, 2)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out

class Mamba2SimpleCat(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=16,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        # others
        local_window=False,
        window_size=7,
        use_pe=False,
        use_norm=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // (self.headdim * 2)
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        assert self.d_inner % 2 == 0
        # Order: [z, x, B, C, dt]
        # d_inner 64 d_state 16 d_model 64 ngroups 1 headdim 16
        # print("d_inner", self.d_inner, "d_state", self.d_state, "d_model", self.d_model, "ngroups", self.ngroups, "headdim", self.headdim, "heads", self.nheads)
        d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner//2 + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        # self.norm = RMSNormGated(self.d_inner//2, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        # self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, elementwise_affine=True, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, elementwise_affine=True, **factory_kwargs) if use_norm else nn.Identity()

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.local_window = local_window
        self.use_pe = use_pe

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner//2, self.d_inner//2 + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        # print(z.shape, xBC.shape, dt.shape)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]

        # 1D Convolution
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, self.d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner//2, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        # y = self.norm(y, z)
        y = torch.cat([y, z], dim=-1)
        y = self.norm(y)
        out = self.out_proj(y)
        return out
    
    
    
class MultipleInputIdentity(nn.Module):
    def __init__(self, final_op=None):
        super().__init__()
        self.final_op = final_op

    def forward(self, x, z):
        if self.final_op == "gate":
            return x * F.silu(z)
        elif self.final_op == "add":
            return x + z
        elif self.final_op == "mul":
            return x * z
        elif self.final_op == "cat":
            return torch.cat([x, z], dim=-1)
        else:
            raise NotImplementedError
    # def forward(self, x, z):
    #     return torch.cat([x, z], dim=-1)

class CrossMamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_model2=None,
        share_conv1d=False,
        norm_layer=None,
        final_op=None,
        d_state=16,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=16,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        # others
        local_window=False,
        window_size=7,
        use_pe=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_model2 = d_model2 if d_model2 is not None else d_model
        self.share_conv1d = share_conv1d
        final_op = "gate" if final_op is None else final_op
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + self.ngroups * self.d_state + self.nheads
        d_in_proj_b = self.ngroups * self.d_state
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        self.in_proj_b = nn.Linear(self.d_model2, d_in_proj_b, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        if self.share_conv1d:
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner + self.ngroups * self.d_state,
                out_channels=self.d_inner + self.ngroups * self.d_state,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner + self.ngroups * self.d_state,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.conv1d_b = nn.Conv1d(
                in_channels=self.ngroups * self.d_state,
                out_channels=self.ngroups * self.d_state,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.ngroups * self.d_state,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
            if not self.share_conv1d:
                nn.init.uniform_(self.conv1d_b.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True


        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        if norm_layer == nn.Identity:
            self.norm = MultipleInputIdentity(final_op=final_op) 
        else:
            if final_op == "cat":
                self.norm = nn.Sequential(
                    MultipleInputIdentity(final_op=final_op),
                    nn.LayerNorm(self.d_inner + self.d_inner, eps=1e-5, elementwise_affine=True, **factory_kwargs)
                )
            else:
                self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        # self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, elementwise_affine=True, **factory_kwargs)

        if final_op == "cat":
            self.out_proj = nn.Linear(self.d_inner + self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.local_window = local_window
        self.use_pe = use_pe
        if self.local_window:
            if self.use_pe == "on_c":
                self.pe = nn.Conv2d(self.ngroups * self.d_state, self.ngroups * self.d_state, kernel_size=3, padding=1, groups=self.ngroups * self.d_state, bias=False)
            else:
                raise NotImplementedError
            self.window_size = window_size

    def forward(self, u, support, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbdt = self.in_proj(u)  # (B, L, d_in_proj)
        c = self.in_proj_b(support)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        z, xB, dt = torch.split(
            zxbdt, [self.d_inner, self.d_inner + self.ngroups * self.d_state, self.nheads], dim=-1
        )
        C = c
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]

        if self.share_conv1d:
            xBC = torch.cat([xB, C], dim=-1)
            # 1D Convolution
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
            )  # (B, L, self.d_inner + 2 * ngroups * d_state)
            xBC = xBC[:, :seqlen, :]
            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        else:
            xB = self.act(
                self.conv1d(xB.transpose(1, 2)).transpose(1, 2)
            )
            xB = xB[:, :seqlen, :]
            C = self.act(
                self.conv1d_b(C.transpose(1, 2)).transpose(1, 2)
            )
            C = C[:, :seqlen, :]
            x, B = torch.split(xB, [self.d_inner, self.ngroups * self.d_state], dim=-1)

        if self.local_window:
            if self.use_pe == "on_c":
                C = self.pe(C.transpose(1, 2).reshape(batch, -1, self.window_size, self.window_size).contiguous()).flatten(2).transpose(1, 2)
            else:
                raise NotImplementedError
            
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 d_state=16,
                 d_conv=3,
                 expand=1,
                 mlp_ratio=2., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 mixer="mamba", # mamba, bimamba
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 local_window=False,
                 window_size=7,
                 num_additional_blocks=0,
                 share_conv1d=True, # for mamba2
                 use_pe=False,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if mixer == "mamba2simple":
            self.mixer = Mamba2Simple(d_model=dim, 
                                          d_state=d_state,
                                          d_conv=d_conv,
                                          expand=expand,
                                          local_window=local_window,
                                          window_size=window_size,
                                          use_pe=use_pe,
                                            )
        elif mixer == "mamba2simplecat":
            self.mixer = Mamba2SimpleCat(d_model=dim, 
                                          d_state=d_state,
                                          d_conv=d_conv,
                                          expand=expand,
                                          local_window=local_window,
                                          window_size=window_size,
                                          use_pe=use_pe,
                                            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        # init mlp weight to 0.1 in mlp
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    
class CrossMambaAlignmentBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 dim2=None,
                 d_state=16,
                 d_conv=3,
                 expand=1,
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 mixer="mamba", # mamba, bimamba
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 local_window=False,
                 window_size=7,
                 num_additional_blocks=0,
                 share_conv1d=True, # for mamba2
                 use_pe=False,
                 ):
        super().__init__()
        self.local_window = local_window
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        dim2 = dim if dim2 is None else dim2
        self.norm1_b = norm_layer(dim2)
        final_op = mixer.split("_")[-1]
        self.mixer = CrossMamba2Simple(d_model=dim, 
                                        d_model2=dim2,
                                        share_conv1d=share_conv1d,
                                        norm_layer=norm_layer,
                                        d_state=d_state,
                                        d_conv=d_conv,
                                        expand=expand,
                                        final_op=final_op,
                                        local_window=local_window,
                                        window_size=window_size,
                                        use_pe=use_pe,
                                        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # Resolve MLP block safely without eval
        _mlp_map = {
            'Mlp': Mlp,
            'SwiGLU': SwiGLU,
            'GatedMlp': GatedMlp,
            'GluMlp': GluMlp,
        }
        if isinstance(Mlp_block, str):
            if Mlp_block not in _mlp_map:
                raise ValueError(f'Unknown Mlp_block: {Mlp_block}')
            MlpBlockCls = _mlp_map[Mlp_block]
        else:
            MlpBlockCls = Mlp_block
        self.mlp = MlpBlockCls(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        self.num_additional_blocks = num_additional_blocks
        if self.num_additional_blocks > 0:
            raise NotImplementedError("Additional blocks not implemented for CrossMambaAlignmentBlock")

    def forward(self, x, y):
        B, C, H, W = x.shape

        if self.local_window:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                y = torch.nn.functional.pad(y, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)
            y = window_partition(y, self.window_size)
        else:
            x = x.flatten(2).transpose(1, 2)
            y = y.flatten(2).transpose(1, 2)

        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x), self.norm1_b(y)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        if self.num_additional_blocks > 0:
            for _, blk in enumerate(self.additional_blocks):
                x = blk(x)

        if self.local_window:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            assert x.shape[2] == C
            x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()

        return x
    
class MambaVisionLayer_Ours(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth=1,
                 num_heads=0,
                 conv=False,
                 downsample=False,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
                 ####
                 mamba_block_version="v1",
                 block_prev_dwconv=False,
                 # 
                 local_window=False,
                 window_size=7,
                 norm_layer=nn.LayerNorm,
                 num_additional_blocks=0,
                 share_conv1d=True,
                 use_pe=False,
                 Mlp_block="Mlp",
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = True
        self.mamba_block_version = mamba_block_version
        self.block_prev_dwconv = block_prev_dwconv
        if self.block_prev_dwconv:
            self.prev_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            if "v1" in mamba_block_version:
                if mamba_block_version == "v1_mamba2simplecat":
                    mixer_type = "mamba2simplecat"
                self.blocks = nn.ModuleList([Block(dim=dim,
                                                   counter=i, 
                                                   transformer_blocks=transformer_blocks,
                                                   num_heads=num_heads,
                                                   mlp_ratio=mlp_ratio,
                                                   qkv_bias=qkv_bias,
                                                   qk_scale=qk_scale,
                                                   drop=drop,
                                                   attn_drop=attn_drop,
                                                   mixer=mixer_type,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale,
                                                   norm_layer=norm_layer, 
                                                   Mlp_block=Mlp_block,
                                                   local_window=local_window,
                                                   window_size=window_size,
                                                   num_additional_blocks=num_additional_blocks,
                                                   share_conv1d=share_conv1d, # for mamba2
                                                   use_pe=use_pe,)
                                                   for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size


    def forward(self, x):
        _, _, H, W = x.shape
        if self.block_prev_dwconv:
            x = x + self.prev_dwconv(x)


        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)
