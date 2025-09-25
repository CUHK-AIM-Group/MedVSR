import datetime
import logging
import math
import time
import torch
from os import path as osp

import archs  # noqa F401
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
# from basicsr.utils.options import copy_opt_file, dict2str, parse_options

import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY
import warnings
import argparse
import os
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, 'r') as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def _postprocess_yml_value(value):
    # None
    if value == '~' or value.lower() == 'none':
        return None
    # bool
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    # !!float number
    if value.startswith('!!float'):
        return float(value.replace('!!float', ''))
    # number
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
        return float(value)
    # list
    if value.startswith('['):
        return eval(value)
    # str
    return value


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--weight', default=None, type=str)
    parser.add_argument('--save_imgs', action='store_true')


    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    # distributed settings
    opt['dist'] = False
    opt['rank'], opt['world_size'] = get_dist_info()

    seed = random.randint(1, 10000)
    opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    opt['is_train'] = False

    opt['num_gpu'] = torch.cuda.device_count()

    results_root = "/".join(args.weight.split("/")[:-1])
    results_root = results_root.replace("/models/", "/visualization/")
    if not osp.exists(results_root):
        os.makedirs(results_root)
    # results_root = osp.join(results_root, opt['name'])
    # if not osp.exists(results_root):
    #     os.makedirs(results_root)

    opt['path']['results_root'] = results_root
    opt['path']['log'] = results_root
    opt['path']['visualization'] = results_root

    opt['val']['save_img'] = args.save_imgs

    return opt, args


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile
    cmd = ' '.join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, 'r+') as f:
        lines = f.readlines()
        lines.insert(0, f'# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n')
        f.seek(0)
        f.writelines(lines)

@DATASET_REGISTRY.register()
class HKRecurrentDataset(data.Dataset):
    """REDS dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_REDS_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    000 100 (720,1280,3)
    001 100 (720,1280,3)
    ...

    Key examples: "000/00000000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        dataroot_flow (str, optional): Data root path for flow.
        meta_info_file (str): Path for meta information file.
        val_partition (str): Validation partition types. 'REDS4' or 'official'.
        io_backend (dict): IO backend type and other kwarg.
        num_frame (int): Window size for input frames.
        gt_size (int): Cropped patched size for gt patches.
        interval_list (list): Interval list for temporal augmentation.
        random_reverse (bool): Random reverse input frames.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(HKRecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                # print([f'{folder}/{i:04d}' for i in range(int(frame_num))])
                self.keys.extend([f'{folder}/{i:04d}' for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['val_partition'] == 'REDS4':
            val_partition = ['000', '011', '015', '020']
        elif opt['val_partition'] == 'official':
            val_partition = [f'{v:03d}' for v in range(240, 270)]
        else:
            raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                             f"Supported ones are ['official', 'REDS4'].")
        if opt['test_mode']:
            self.keys = [v for v in self.keys if v.split('/')[0] in val_partition]
        else:
            self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000
        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        # print("interval", interval, self.interval_list)
        # print(interval, self.interval_list)

        # ensure not exceeding the borders
        # print("frame_name", frame_name)
        start_frame_idx = int(frame_name)
        # print("start_frame_idx", start_frame_idx)
        if start_frame_idx > 50 - self.num_frame * interval:
            # print("start_frame_idx is greater than 50, self.num_frame: {}, interval: {}".format(self.num_frame, interval))
            # print(50 - self.num_frame * interval)
            start_frame_idx = random.randint(0, 50 - self.num_frame * interval)
            # print("start_frame_idx", start_frame_idx)

        end_frame_idx = start_frame_idx + self.num_frame * interval
        # print("end_frame_idx", end_frame_idx)

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        # print("neighbor_list", neighbor_list)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()


        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:04d}'
                img_gt_path = f'{clip_name}/{neighbor:04d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:04d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:04d}.png'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        # print("obtained data")
        # print({'lq': img_lqs.shape, 'gt': img_gts.shape, 'key': key})
        return {'lq': img_lqs, 'gt': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
    

def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    rank, _ = get_dist_info()
    dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)

def create_train_val_dataloader(opt):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase.split('_')[0] == 'val':
            dataset_opt['dataroot_gt'] = dataset_opt['dataroot_gt'].replace('val', 'test')
            dataset_opt['dataroot_lq'] = dataset_opt['dataroot_lq'].replace('val', 'test')
            print("dataset_opt", dataset_opt)
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            print(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        elif phase == 'train':
            pass
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return val_loaders


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
        
    # copy the yml file to the experiment root
    if not osp.exists(opt['path']['experiments_root']):
        os.makedirs(opt['path']['experiments_root'])
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)
    for dataset in [
                    "hyperkvasir", 
                    "lppolyp",
                    "endovis",
                    ]:
        val_loaders = []
        for phase, dataset_opt in opt['datasets'].items():
            if phase.split('_')[0] == 'val':
                if dataset == "hyperkvasir":
                    opt['path']['visualization'] = os.path.join(opt['path']['visualization'], "hyperkvasir")
                    dataset_opt['dataroot_gt'] = "./hyperkvasir_test/GT"
                    dataset_opt['dataroot_lq'] = "./hyperkvasir_test/BIx4"
                elif dataset == "lppolyp":
                    opt['path']['visualization'] = os.path.join(opt['path']['visualization'], "lppolyp")
                    dataset_opt['dataroot_gt'] = "./ldpolyp_test/GT"
                    dataset_opt['dataroot_lq'] = "./ldpolyp_test/BIx4"
                elif dataset == "endovis":
                    opt['path']['visualization'] = os.path.join(opt['path']['visualization'], "endovis")
                    dataset_opt['dataroot_gt'] = "./endovis18_test/GT"
                    dataset_opt['dataroot_lq'] = "./endovis18_test/BIx4"
                elif dataset == "cartar":
                    opt['path']['visualization'] = os.path.join(opt['path']['visualization'], "cartar")
                    dataset_opt['dataroot_gt'] = "./cataract_101/GT"
                    dataset_opt['dataroot_lq'] = "./cataract_101/BIx4"
                else:
                    raise ValueError(f"Dataset {dataset} is not recognized.")
                print("dataset_opt", dataset_opt)
                val_set = build_dataset(dataset_opt)
                val_loader = build_dataloader(
                    val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
                print(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
                val_loaders.append(val_loader)

        # share the same validation function with VideoRecurrentModel
        opt['model_type'] = "VideoRecurrentModel" if opt['model_type'] == "RecurrentMixPrecisionRTModel" else opt['model_type'] 
        model = build_model(opt)
        model.print_model_info()
        for para_dict in [
                            "params", 
                            ]:
            print("Loading weights from", args.weight, para_dict)
            ckpt = torch.load(args.weight)[para_dict]
            try:
                model.net_g.load_state_dict(ckpt, strict=True)
            except:
                ckpt_new = OrderedDict()
                for k, v in ckpt.items():
                    if not k.startswith("module."):
                        k = "module." + k
                    ckpt_new[k] = v
                model.net_g.load_state_dict(ckpt_new, strict=True)
            for val_loader in val_loaders:
                model.validation(val_loader, 0, tb_logger, opt['val']['save_img'])

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print(root_path)
    train_pipeline(root_path)
