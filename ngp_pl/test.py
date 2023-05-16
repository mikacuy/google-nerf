import torch
import time
import numpy as np
from models.networks import NGP
from models.rendering import render
from metrics import psnr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import dataset_dict
from utils import load_ckpt
from train import depth2img
from datasets.ray_utils import axisangle_to_R, get_rays

scene = 'Lego'
dataset = dataset_dict['nsvf'](
    f'/orion/group/NSVF/Synthetic_NeRF/{scene}/',
    split='test', downsample=1.0
)

model = NGP(scale=0.5).cuda()
load_ckpt(model, f'ckpts/nsvf/{scene}/epoch=29_slim.ckpt')

psnrs = []; ts = []

for img_idx in tqdm(range(len(dataset))):
    rays = dataset.rays[img_idx][:, :6].cuda()
    
    poses = dataset.poses[img_idx].cuda()
    
    rays_o, rays_d = get_rays(dataset.directions.cuda(), poses)

    t = time.time()
    results = render(model, rays_o, rays_d, **{'test_time': True, 'T_threshold': 1e-2})
    
    torch.cuda.synchronize()
    ts += [time.time()-t]

    if dataset.split != 'test_traj':
        
#         rgb_gt = dataset.rays[img_idx][:, 6:].cuda()
        rgb_gt = dataset.rays[img_idx].cuda()
        psnrs += [psnr(results['rgb'], rgb_gt).item()]
        
if psnrs: print(f'mean PSNR: {np.mean(psnrs):.2f}, min: {np.min(psnrs)}, max: {np.max(psnrs)}')
print(f'mean time: {np.mean(ts):.4f} FPS: {1/np.mean(ts):.2f}')