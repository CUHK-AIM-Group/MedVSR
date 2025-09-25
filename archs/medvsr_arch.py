import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import warnings

from basicsr.archs.arch_util import flow_warp
from torch.nn import init as init
# from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.archs.spynet_arch import SpyNet
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn.modules.batchnorm import _BatchNorm

from .modules.mamba_vision import MambaVisionLayer_Ours, CrossMambaAlignmentBlock
from timm.layers.mlp import SwiGLU, GatedMlp, GluMlp
from timm.models.vision_transformer import Mlp as TimmMlp
from timm.models.layers import LayerNorm2d
# SE


from functools import partial
from typing import Optional, Callable

from mamba_ssm.modules.mamba_simple import Mamba
        
    

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False, act='relu', expand_ratio=1, dw=False, dw_version="v1", kernel_size=3):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        inner_feat = int(num_feat * expand_ratio)
        if dw:
            if "v1" in dw_version:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(num_feat, inner_feat, kernel_size, 1, padding=kernel_size//2, groups=num_feat, bias=True),
                    nn.Conv2d(inner_feat, num_feat, 1, 1, 0, groups=1, bias=True),
                    SEModule(inner_feat, 16) if "se" in dw_version else nn.Identity(),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(num_feat, inner_feat, kernel_size, 1, padding=kernel_size//2, groups=num_feat, bias=True),
                    nn.Conv2d(inner_feat, num_feat, 1, 1, 0, groups=1, bias=True),
                )
            else:
                raise ValueError(f"dw_version={dw_version} is not supported.")
        else:
            self.conv1 = nn.Conv2d(num_feat, inner_feat, 3, 1, 1, groups=1, bias=True)
            self.conv2 = nn.Conv2d(inner_feat, num_feat, 3, 1, 1, groups=1, bias=True)
        if act == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif act == 'leakyrelu':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act == 'relu6':
            self.relu = nn.ReLU6(inplace=True)
        elif act == 'gelu':
            self.relu = nn.GELU()
        else:
            raise ValueError(f'act={act} is not supported.')


        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15, act='relu', expand_ratio=1, dw=False, dw_version="v1"):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch, act=act, expand_ratio=expand_ratio, dw=dw, dw_version=dw_version))

    def forward(self, fea):
        return self.main(fea)


class HybridConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15, act='relu', expand_ratio=1, n_mambablocks=0, conv_first=True, dw=False,
                    mamba_block_version="v1", block_prev_dwconv=False,
                    local_window=False, window_size=7,
                    norm_layer=nn.LayerNorm,
                    use_pe="none", Mlp_block="Mlp", num_mid_ch=None,
                    progressive=False,
                    dw_version="v1",
                    ks=3):
        super().__init__()
        n_resblocks = num_block - n_mambablocks
        num_mid_ch = num_out_ch if num_mid_ch is None else num_mid_ch
        if n_mambablocks != 0:
            self.main = nn.Sequential(
                nn.Conv2d(num_in_ch, num_mid_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                make_layer(MambaVisionLayer_Ours, n_mambablocks, dim=num_mid_ch, mamba_block_version=mamba_block_version, block_prev_dwconv=block_prev_dwconv, 
                        local_window=local_window, window_size=window_size, norm_layer=norm_layer,
                        use_pe=False, Mlp_block=Mlp_block),
                make_layer(ResidualBlockNoBN, n_resblocks, num_feat=num_mid_ch, act=act, expand_ratio=expand_ratio, dw=dw, dw_version=dw_version, kernel_size=ks),
                nn.Identity() if num_mid_ch == num_out_ch else nn.Conv2d(num_mid_ch, num_out_ch, 3, 1, 1, groups=num_mid_ch if dw else 1, bias=True),
                )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch, act=act, expand_ratio=expand_ratio, dw=dw, dw_version=dw_version))

    def forward(self, fea):
        return self.main(fea)


@ARCH_REGISTRY.register()
class MedVSR(nn.Module):

    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 max_residue_magnitude=10,
                 is_low_res_input=True,
                 spynet_path=None,
                 cpu_cache_length=100,
                 ### conv
                 conv_type='normal', # normal, expanded
                 conv_act='relu',
                 ### add mamba in
                 n_mambablocks=0,
                 conv_first=True,
                 mamba_block_version="v1",
                 block_prev_dwconv=False,
                 # ablate number of blocks
                 n_convfe = 5,
                 n_convrec = 5,
                 fe_type='hybrid', # normal, hybrid, mamba, hrepscale
                 fe_dw=False,
                 fe_dw_version="v1",
                 fe_expand_ratio=1,
                 trunk_type='hybrid', # normal, hybrid, mamba, hrepscale, stack
                 trunk_dw=False,
                 trunk_dw_version="v1",
                 trunk_expand_ratio=1,
                 rec_type='hybrid', # normal, hybrid, mamba, hrepscale
                 rec_dw=False,
                 rec_expand_ratio=1,
                 rec_dw_version="v1",
                 rec_ks=3,
                 # for alignment
                 flow_warp=True,
                 deform_align_or_not=True,
                 deform_align_method="default", # default, modify1, modify2
                 deform_align_duo_flow_n1=False,
                 deform_align_use_feat_cur_n1_n2="n2",
                 cross_arc_shared=True,
                 cross_arc_order="both",
                 cross_arc="none",
                 local_window=False,
                 window_size=7,
                 use_pe="none",
                 arc_mixer="mamba",
                 num_additional_blocks=0,
                 share_conv1d=True,
                 norm_layer=nn.LayerNorm,
                 Mlp_block="Mlp",
                 # misc propagation configs
                 prop_progressive=False,
                 interp_mode='bilinear',
                 # 
                 layer_scale=None,
                 # 
                 use_previous_branch_features=False,
                 ):

        super().__init__()
        self.mid_channels = mid_channels
        self.cross_arc = cross_arc
        self.is_low_res_input = is_low_res_input
        self.cpu_cache_length = cpu_cache_length

        self.flow_warp = flow_warp
        self.deform_align_or_not = deform_align_or_not
        
        self.cross_arc_shared = cross_arc_shared
        self.cross_arc_order = cross_arc_order
        self.prop_progressive = prop_progressive
        self.interp_mode = interp_mode
        self.trunk_type = trunk_type

        self.deform_align_method = deform_align_method
        self.deform_align_duo_flow_n1=deform_align_duo_flow_n1
        self.deform_align_use_feat_cur_n1_n2 = deform_align_use_feat_cur_n1_n2

        self.use_previous_branch_features = use_previous_branch_features

        # optical flow
        self.spynet = SpyNet(spynet_path)

        # feature extraction module
        if is_low_res_input:
            if fe_type == 'normal':
                self.feat_extract = ConvResidualBlocks(3, mid_channels, n_convfe, act=conv_act, expand_ratio=fe_expand_ratio, dw=fe_dw, dw_version=fe_dw_version)
        else:
            if fe_type == 'normal':
                self.feat_extract = nn.Sequential(
                    nn.Conv2d(3, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Conv2d(mid_channels, mid_channels, 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    ConvResidualBlocks(mid_channels, mid_channels, n_convfe, act=conv_act, expand_ratio=fe_expand_ratio, dw=fe_dw, dw_version=fe_dw_version))


        # propagation branches
        if self.deform_align_or_not:
            self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            if not self.prop_progressive:
                if self.deform_align_or_not:
                    if torch.cuda.is_available():
                        if self.deform_align_method == "default":
                            deform_alignblock = SecondOrderDeformableAlignment
                        else:
                            raise ValueError(f"deform_align_method={self.deform_align_method} is not supported.")
                        self.deform_align[module] = deform_alignblock(
                            2 * mid_channels,
                            mid_channels,
                            3,
                            padding=1,
                            deformable_groups=16,
                            max_residue_magnitude=max_residue_magnitude)
                if trunk_type == 'normal':
                    self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks, act=conv_act, expand_ratio=trunk_expand_ratio, dw=trunk_dw, dw_version=trunk_dw_version)

        # helper resolvers
        def _resolve_norm_layer(nl):
            if isinstance(nl, str):
                mapping = {
                    'LayerNorm': nn.LayerNorm,
                    'nn.LayerNorm': nn.LayerNorm,
                    'LayerNorm2d': LayerNorm2d,
                    'nn.BatchNorm2d': nn.BatchNorm2d,
                    'BatchNorm2d': nn.BatchNorm2d,
                }
                if nl in mapping:
                    return mapping[nl]
                raise ValueError(f'Unknown norm_layer string: {nl}')
            return nl

        def _resolve_mlp_block(mb):
            if isinstance(mb, str):
                mapping = {
                    'Mlp': TimmMlp,
                    'SwiGLU': SwiGLU,
                    'GatedMlp': GatedMlp,
                    'GluMlp': GluMlp,
                }
                if mb in mapping:
                    return mapping[mb]
                raise ValueError(f'Unknown Mlp_block string: {mb}')
            return mb

        if self.cross_arc_shared:
            self.cross_arc_block = CrossMambaAlignmentBlock(mid_channels, 
                                                            local_window=local_window, 
                                                            window_size=window_size, 
                                                            mixer=arc_mixer, 
                                                            num_additional_blocks=num_additional_blocks,
                                                            share_conv1d=share_conv1d,
                                                            use_pe=use_pe,
                                                            norm_layer=_resolve_norm_layer(norm_layer),
                                                            Mlp_block=_resolve_mlp_block(Mlp_block),
                                                            layer_scale=layer_scale)
        # upsampling module
        self.reconstruction = HybridConvResidualBlocks(5 * mid_channels, mid_channels, n_convrec, act=conv_act, expand_ratio=rec_expand_ratio,
                                                                    n_mambablocks=n_mambablocks, conv_first=conv_first, dw=rec_dw,
                                                                    mamba_block_version=mamba_block_version, block_prev_dwconv=block_prev_dwconv,
                                                                    local_window=local_window, window_size=window_size, norm_layer=_resolve_norm_layer(norm_layer),
                                                                    use_pe=use_pe,
                                                                    Mlp_block=_resolve_mlp_block(Mlp_block),
                                                                    dw_version=rec_dw_version,
                                                                    ks=rec_ks)

        self.upconv1 = nn.Conv2d(mid_channels, mid_channels * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(mid_channels, 64 * 4, 3, 1, 1, bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        if len(self.deform_align) > 0:
            self.is_with_alignment = True
        else:
            self.is_with_alignment = False
            warnings.warn('Deformable alignment module is not added. '
                          'Probably your CUDA is not configured correctly. DCN can only '
                          'be used with CUDA enabled. Alignment is skipped now.')

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the flows used for forward-time propagation \
                (current to previous). 'flows_backward' corresponds to the flows used for backward-time \
                propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = flows_backward.flip(1)
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated \
                features. Each key in the dictionary corresponds to a \
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            if not self.use_previous_branch_features:
                feat_current = feats['spatial'][mapping_idx[idx]]
            else:
                raise NotImplementedError("use_previous_branch_features is not implemented yet.")
                
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0 and self.is_with_alignment:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1), interp_mode=self.interp_mode) if self.flow_warp else feat_prop
                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))

                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1), interp_mode=self.interp_mode) if self.flow_warp else feat_n2
                    cond_n2 = self.cross_arc_block(feat_prop, cond_n2)

                if i == 1:
                    if not self.deform_align_or_not:
                        cond_n2 = cond_n1

                # flow-guided deformable convolution
                if self.deform_align_or_not:
                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_prop], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n1)
                else:
                    feat_prop = cond_n2

            if not self.prop_progressive:
                feat = [feat_current] + [feats[k][idx] for k in feats if k not in ['spatial', module_name]] + [feat_prop]
                if self.cpu_cache:
                    feat = [f.cuda() for f in feat]

                feat = torch.cat(feat, dim=1)

                feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        outputs = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            if not self.prop_progressive:
                hr = [feats[k].pop(0) for k in feats if k != 'spatial']
                hr.insert(0, feats['spatial'][mapping_idx[i]])
                hr = torch.cat(hr, dim=1)
                if self.cpu_cache:
                    hr = hr.cuda()
            else:
                hr = feats['forward_2'][i]
                if self.cpu_cache:
                    hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, c, h, w = lqs.size()

        # whether to cache the features in CPU
        self.cpu_cache = True if t > self.cpu_cache_length else False

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25, mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propgation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'

                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)

    
class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)
