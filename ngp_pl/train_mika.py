import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict

from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils import slim_ckpt

import warnings; warnings.filterwarnings("ignore")

import time

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


### Args
hparams = get_opts()
if hparams.val_only and (not hparams.ckpt_path):
    raise ValueError('You need to provide a @ckpt_path for validation!')

device = "cuda"

### NeRF hyperparameters
S = 16
loss_func = NeRFLoss()
train_psnr = PeakSignalNoiseRatio(data_range=1)
val_psnr = PeakSignalNoiseRatio(data_range=1)
val_ssim = StructuralSimilarityIndexMeasure(data_range=1)

if hparams.eval_lpips:
    val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
    for p in val_lpips.net.parameters():
        p.requires_grad = False

model = NGP(scale=hparams.scale)
G = model.grid_size
model.register_buffer('density_grid',
    torch.zeros(model.cascades, G**3))
model.register_buffer('grid_coords',
    create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
model.to(device, dtype=torch.float)

### Dataset Inits
dataset = dataset_dict[hparams.dataset_name]
kwargs = {'root_dir': hparams.root_dir,
          'downsample': hparams.downsample}
train_dataset = dataset(split=hparams.split, **kwargs)
train_dataset.batch_size = hparams.batch_size

test_dataset = dataset(split='test', **kwargs)

train_dataloader = DataLoader(train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

test_dataloader = DataLoader(test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

directions = train_dataset.directions.to(device)
train_poses = train_dataset.poses.to(device)

print(directions.shape)
print(train_poses.shape)
print(test_dataset.poses.shape)
print()

### Optimizers --> note here there is no optimization for extrinsics yet
net_params = model.parameters()


net_opt = FusedAdam(net_params, hparams.lr, eps=1e-15)

net_sch = CosineAnnealingLR(net_opt,
                            hparams.num_epochs,
                            hparams.lr/30)


model.mark_invisible_cells(train_dataset.K.to(device),
                                        train_poses,
                                        train_dataset.img_wh)


model.train()
global_step = 0

for epoch in range(hparams.num_epochs):
    for batch in train_dataloader:

        ### Move to cuda
        batch["rgb"] = batch["rgb"].to(device)
        batch["img_idxs"] = batch["img_idxs"].to(device)

        if global_step%S == 0:
            model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=global_step<256,
                                           erode=hparams.dataset_name=='colmap')        

        poses = train_poses[batch['img_idxs']]
        rays_o, rays_d = get_rays(directions[batch['pix_idxs']], poses)

        kwargs = {'test_time': False}
        if hparams.dataset_name in ['colmap', 'nerfpp']:
            kwargs['exp_step_factor'] = 1/256   


        results = render(model, rays_o, rays_d, **kwargs)

        loss_d = loss_func(results, batch)


        loss = sum(lo.mean() for lo in loss_d.values())

        print(loss)
        print(loss.dtype)

        net_opt.zero_grad()
        print("zero grad")
        loss.backward()
        print("backward")
        net_opt.step()
        print("opt step")

        net_sch.step()
        print("sched step")

        global_step += 1

        print(loss)
        print(results['total_samples']/len(rays_o))

        ### Logging
        print("PSNR")
        with torch.no_grad():
            print(train_psnr(results['rgb'], batch['rgb']))
        # print(psnr)

        exit()















































