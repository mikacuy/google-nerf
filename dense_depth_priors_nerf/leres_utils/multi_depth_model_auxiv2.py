import torch
import torch.nn.functional
import torch.nn as nn

from . import network_auxi as network
from . import net_tools

class RelDepthModel(nn.Module):
    def __init__(self):
        super(RelDepthModel, self).__init__()
        self.depth_model = DepthModel()
        # self.losses = ModelLoss()

    def forward(self, data, is_train=True):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        self.inputs = data['rgb'].cuda()
        self.logit, self.auxi = self.depth_model(self.inputs)

        # if is_train:
        #     self.losses_dict = self.losses.criterion(self.logit, self.auxi, data)
        # else:
        #     self.losses_dict = {'total_loss': torch.tensor(0.0, dtype=torch.float).cuda()}
        self.losses_dict = None

        return {'decoder': self.logit, 'auxi': self.auxi, 'losses': self.losses_dict}

    def inference(self, data, return_loss=False):
        # with torch.no_grad():
        #     out = self.forward(data, is_train=False)
        #     pred_depth = out['decoder']
        #     pred_disp = out['auxi']
        #     pred_depth_normalize = (pred_depth - pred_depth.min() + 1) / (pred_depth.max() - pred_depth.min()) #pred_depth - pred_depth.min() #- pred_depth.max()
        #     pred_depth_out = pred_depth
        #     pred_disp_normalize = (pred_disp - pred_disp.min() + 1) / (pred_disp.max() - pred_disp.min())
        #     return {'pred_depth': pred_depth_out, 'pred_depth_normalize': pred_depth_normalize,
        #             'pred_disp': pred_disp, 'pred_disp_normalize': pred_disp_normalize,
        #             }

        self.inputs = data['rgb'].cuda()
        depth, auxi = self.depth_model(self.inputs)
        pred_depth_out = depth
        pred_depth_out = depth - depth.min() + 0.01

        if return_loss:
            losses_dict = self.losses.criterion(depth, auxi, data)
            return pred_depth_out, losses_dict

        return pred_depth_out

#### To incorporate cIMLE
class RelDepthModel_cIMLE(nn.Module):
    def __init__(self, d_latent=512, version="v2"):
        super(RelDepthModel_cIMLE, self).__init__()
        self.depth_model = DepthModel_cIMLE(d_latent=d_latent, version=version)
        # self.losses = ModelLoss()

    def forward(self, data, z, is_train=True, transform_pred=False, scale=1.0, shift=0.0):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real

        data = data.cuda()
        z = z.cuda()

        self.inputs = data
        self.logit = self.depth_model(self.inputs, z)

        ## Scale so that it is non-negative
        self.logit = self.logit - self.logit.min() + 0.01

        self.auxi = None

        # if is_train:
        #     self.losses_dict = self.losses.criterion(self.logit, self.auxi, data)

        # else:
        #     self.losses_dict = {'total_loss': torch.tensor(0.0, dtype=torch.float).cuda()}

        self.losses_dict = None

        return {'decoder': self.logit, 'auxi': self.auxi, 'losses': self.losses_dict}

    def inference(self, data, z, rescaled=False):
        # with torch.no_grad():
        #     out = self.forward(data, is_train=False)
        #     pred_depth = out['decoder']
        #     pred_disp = out['auxi']
        #     pred_depth_normalize = (pred_depth - pred_depth.min() + 1) / (pred_depth.max() - pred_depth.min()) #pred_depth - pred_depth.min() #- pred_depth.max()
        #     pred_depth_out = pred_depth
        #     pred_disp_normalize = (pred_disp - pred_disp.min() + 1) / (pred_disp.max() - pred_disp.min())
        #     return {'pred_depth': pred_depth_out, 'pred_depth_normalize': pred_depth_normalize,
        #             'pred_disp': pred_disp, 'pred_disp_normalize': pred_disp_normalize,
        #             }

        data['rgb'] = data['rgb'].cuda()
        z = z.cuda()

        self.inputs = data['rgb']
        depth = self.depth_model(self.inputs, z)
        pred_depth_out = depth

        if rescaled:  
            pred_depth_out = depth - depth.min() + 0.01

        return pred_depth_out

    def set_mean_var_shifts(self, mean0, var0, mean1, var1, mean2, var2, mean3, var3):
        return self.depth_model.set_mean_var_shifts(mean0, var0, mean1, var1, mean2, var2, mean3, var3)

    def get_adain_init_act(self, data, z):
        data['rgb'] = data['rgb'].cuda()
        z = z.cuda()

        self.inputs = data['rgb']
        return self.depth_model.get_adain_init_act(self.inputs, z)


### Incorporate cIMLE to decoder
class RelDepthModel_cIMLE_decoder(nn.Module):
    def __init__(self, d_latent=512, version="v2"):
        super(RelDepthModel_cIMLE_decoder, self).__init__()
        self.depth_model = DepthModel_cIMLE_v2(d_latent=d_latent, version=version)
        # self.losses = ModelLoss()

    def forward(self, data, z, is_train=True, transform_pred=False, scale=1.0, shift=0.0):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real

        data['rgb'] = data['rgb'].cuda()
        z = z.cuda()

        self.inputs = data['rgb']
        self.logit = self.depth_model(self.inputs, z)
        self.auxi = None

        # if is_train:
        #     self.losses_dict = self.losses.criterion(self.logit, self.auxi, data)

        # else:
        #     self.losses_dict = {'total_loss': torch.tensor(0.0, dtype=torch.float).cuda()}
        self.losses_dict = None

        return {'decoder': self.logit, 'auxi': self.auxi, 'losses': self.losses_dict}

    def inference(self, data, z, rescaled=False, return_loss=False):
        # with torch.no_grad():
        #     out = self.forward(data, is_train=False)
        #     pred_depth = out['decoder']
        #     pred_disp = out['auxi']
        #     pred_depth_normalize = (pred_depth - pred_depth.min() + 1) / (pred_depth.max() - pred_depth.min()) #pred_depth - pred_depth.min() #- pred_depth.max()
        #     pred_depth_out = pred_depth
        #     pred_disp_normalize = (pred_disp - pred_disp.min() + 1) / (pred_disp.max() - pred_disp.min())
        #     return {'pred_depth': pred_depth_out, 'pred_depth_normalize': pred_depth_normalize,
        #             'pred_disp': pred_disp, 'pred_disp_normalize': pred_disp_normalize,
        #             }

        data['rgb'] = data['rgb'].cuda()
        z = z.cuda()

        self.inputs = data['rgb']
        depth = self.depth_model(self.inputs, z)
        pred_depth_out = depth

        if rescaled:  
            pred_depth_out = depth - depth.min() + 0.01

        if return_loss:
            losses_dict = self.losses.criterion(depth, None, data)
            return pred_depth_out, losses_dict

        return pred_depth_out

    def set_mean_var_shifts(self, mean0, var0, mean1, var1, mean2, var2, mean3, var3):
        return self.depth_model.set_mean_var_shifts(mean0, var0, mean1, var1, mean2, var2, mean3, var3)

    def get_adain_init_act(self, data, z):
        data['rgb'] = data['rgb'].cuda()
        z = z.cuda()

        self.inputs = data['rgb']
        return self.depth_model.get_adain_init_act(self.inputs, z)


##########################

class ModelOptimizer_AdaIn(object):
    def __init__(self, model, base_lr, mlp_lr, fixed_backbone=False):
        super(ModelOptimizer_AdaIn, self).__init__()
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        mlp_params = []
        mlp_params_names = []

        for key, value in model.named_parameters():
            if value.requires_grad:
                if "style" in key:
                    mlp_params.append(value)
                    mlp_params_names.append(key)
                elif 'encoder' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)


        lr_encoder = base_lr
        lr_decoder = base_lr
        lr_mlp = mlp_lr

        weight_decay = 0.0005

        if not fixed_backbone:
            print("Joint backbone.")
            net_params = [
                {'params': encoder_params,
                 'lr': lr_encoder,
                 'weight_decay': weight_decay},
                {'params': decoder_params,
                 'lr': lr_decoder,
                 'weight_decay': weight_decay},
                {'params': mlp_params,
                 'lr': lr_mlp,
                 'weight_decay': weight_decay},
            ]
        else:
            print("Fixed backbone.")
            net_params = [
                {'params': mlp_params,
                 'lr': lr_mlp,
                 'weight_decay': weight_decay},
            ]

        self.optimizer = torch.optim.SGD(net_params, momentum=0.9)
        self.model = model

    def optim(self):
        self.optimizer.zero_grad()
        # loss_all = loss['total_loss']

        # loss_all = torch.mean(loss_all)

        # loss_all.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + 'resnext101_stride32x8d'

        print(backbone)

        self.encoder_modules = net_tools.get_func(backbone)()
        self.decoder_modules = network.Decoder()
        self.auxi_modules = network.AuxiNetV2()

    def forward(self, x):
        lateral_out = self.encoder_modules(x)

        out_logit, auxi_input = self.decoder_modules(lateral_out)
        out_auxi = self.auxi_modules(auxi_input)
        
        return out_logit, out_auxi

class DepthModel_cIMLE(nn.Module):
    def __init__(self, d_latent=512, version="v2"):
        super(DepthModel_cIMLE, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + 'resnext101_stride32x8d'

        self.encoder_modules = net_tools.get_func(backbone)(cIMLE=True, d_latent=d_latent, version=version)
        self.decoder_modules = network.Decoder()
        # self.auxi_modules = network.AuxiNetV2()

    def forward(self, x, z):
        # print("=========Image size===========")
        # print(x.shape)

        lateral_out = self.encoder_modules(x, z)
        
        # print("=========Depth Model===========")

        # out_logit, auxi_input = self.decoder_modules(lateral_out)
        
        out_logit = self.decoder_modules(lateral_out, auxi=False)
        # print(out_logit.shape)

        # out_auxi = self.auxi_modules(auxi_input)
        # print(out_auxi.shape)
        # exit()
        
        return out_logit

    def set_mean_var_shifts(self, mean0, var0, mean1, var1, mean2, var2, mean3, var3):
        return self.encoder_modules.set_mean_var_shifts(mean0, var0, mean1, var1, mean2, var2, mean3, var3)

    def get_adain_init_act(self, x, z):
        return self.encoder_modules.get_adain_init_act(x, z)


class DepthModel_cIMLE_v2(nn.Module):
    def __init__(self, d_latent=512, version="v2"):
        super(DepthModel_cIMLE_v2, self).__init__()
        backbone = network.__name__.split('.')[-1] + '.' + 'resnext101_stride32x8d'
        self.version = version

        # print(backbone)

        self.encoder_modules = net_tools.get_func(backbone)()

        if self.version in ["v2", "v3","v4","v5","v6"]:
            self.decoder_modules = network.Decoder_cIMLE(d_latent=d_latent, version=version)
        else:
            print("Unimplemented in DepthModel_cIMLE_v2.")
            exit()

        # self.auxi_modules = network.AuxiNetV2()

    def forward(self, x, z):
        # print("=========Image size===========")
        # print(x.shape)

        lateral_out = self.encoder_modules(x)
        
        if self.version == "v2":
            out_logit = self.decoder_modules(lateral_out, z, auxi=False)
        elif self.version in ["v3", "v4","v5","v6"]:
            out_logit = self.decoder_modules(lateral_out, z, x, auxi=False)

        
        return out_logit

    def set_mean_var_shifts(self, mean0, var0, mean1, var1, mean2, var2, mean3, var3):
        return self.decoder_modules.set_mean_var_shifts(mean0, var0, mean1, var1, mean2, var2, mean3, var3)

    def get_adain_init_act(self, x, z):
        lateral_out = self.encoder_modules(x)

        if self.version == "v2":
            return self.decoder_modules.get_adain_init_act(lateral_out, z)
        elif self.version in ["v3", "v4","v5","v6"]:
            return self.decoder_modules.get_adain_init_act(lateral_out, z, x)


def recover_scale_shift_depth(pred, gt, min_threshold=1e-8, max_threshold=1e8):
    b, c, h, w = pred.shape
    mask = (gt > min_threshold) & (gt < max_threshold)  # [b, c, h, w]
    EPS = 1e-6 * torch.eye(2, dtype=pred.dtype, device=pred.device)
    scale_shift_batch = []
    ones_img = torch.ones((1, h, w), dtype=pred.dtype, device=pred.device)
    for i in range(b):
        mask_i = mask[i, ...]
        pred_valid_i = pred[i, ...][mask_i]
        ones_i = ones_img[mask_i]
        pred_valid_ones_i = torch.stack((pred_valid_i, ones_i), dim=0)  # [c+1, n]
        A_i = torch.matmul(pred_valid_ones_i, pred_valid_ones_i.permute(1, 0))  # [2, 2]
        A_inverse = torch.inverse(A_i + EPS)

        gt_i = gt[i, ...][mask_i]
        B_i = torch.matmul(pred_valid_ones_i, gt_i)[:, None]  # [2, 1]
        scale_shift_i = torch.matmul(A_inverse, B_i)  # [2, 1]
        scale_shift_batch.append(scale_shift_i)
    scale_shift_batch = torch.stack(scale_shift_batch, dim=0)  # [b, 2, 1]
    ones = torch.ones_like(pred)
    pred_ones = torch.cat((pred, ones), dim=1)  # [b, 2, h, w]
    pred_scale_shift = torch.matmul(pred_ones.permute(0, 2, 3, 1).reshape(b, h * w, 2), scale_shift_batch)  # [b, h*w, 1]
    pred_scale_shift = pred_scale_shift.permute(0, 2, 1).reshape((b, c, h, w))
    return pred_scale_shift




    
