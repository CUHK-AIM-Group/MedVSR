import torch
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
from basicsr.models.crop_validation import forward_crop


@MODEL_REGISTRY.register()
class EDVRModel(VideoBaseModel):
    """EDVR Model.

    Paper: EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.  # noqa: E501
    """

    def __init__(self, opt):
        super(EDVRModel, self).__init__(opt)
        if self.is_train:
            self.train_tsa_iter = opt['train'].get('tsa_iter')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        dcn_lr_mul = train_opt.get('dcn_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for dcn with {dcn_lr_mul}.')
        if dcn_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate dcn params and normal params for different lr
            normal_params = []
            dcn_params = []
            for name, param in self.net_g.named_parameters():
                if 'dcn' in name:
                    dcn_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': dcn_params,
                    'lr': train_opt['optim_g']['lr'] * dcn_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.train_tsa_iter:
            if current_iter == 1:
                logger = get_root_logger()
                logger.info(f'Only train TSA module for {self.train_tsa_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'fusion' not in name:
                        param.requires_grad = False
            elif current_iter == self.train_tsa_iter:
                logger = get_root_logger()
                logger.warning('Train all the parameters.')
                for param in self.net_g.parameters():
                    param.requires_grad = True

        super(EDVRModel, self).optimize_parameters(current_iter)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            if "crop" in self.opt and self.opt["crop"]:
                # if self.opt['train'].get('flow_opt'):
                #     self.output = forward_crop(self.lq, self.net_g, flow_opt=self.opt['train'].get('flow_opt'))
                # else:
                if 'train' in self.opt['datasets']:
                    lq_size = self.opt['datasets']['train']['gt_size'] // 4
                if 'test' in self.opt['datasets']:
                    lq_size = self.opt['datasets']['test']['lq_size']
                overlap = lq_size // 2   # TODO
                self.output = forward_crop(self.lq, self.net_g, lq_size=lq_size, overlap=overlap)
            else:
                # if self.opt['train'].get('flow_opt'):
                #     self.output, _ = self.net_g(self.lq)
                # else:
                self.output = self.net_g(self.lq)
        self.net_g.train()