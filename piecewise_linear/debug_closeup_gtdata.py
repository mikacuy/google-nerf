import os

import numpy as np
import json
import cv2

import torchvision.transforms as transforms
import imageio
import torch


# fname = "nerf_synthetic/lego_0.5_1.25/./test/r_0.png"
fname = "nerf_synthetic/lego/./test/r_0.png"

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def read_files(rgb_file, downsample_scale=None):
    # fname = os.path.join(basedir, rgb_file)
    fname = rgb_file
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

    if downsample_scale is not None:
        img = cv2.resize(img, (int(img.shape[1]/downsample_scale), int(img.shape[0]/downsample_scale)), interpolation=cv2.INTER_LINEAR)

    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available

    return img


img = read_files(fname)
# cv2.imwrite("closeup_gt_loaded.png", cv2.cvtColor(to8b(img), cv2.COLOR_RGB2BGR))
cv2.imwrite("original_gt_loaded.png", cv2.cvtColor(to8b(img), cv2.COLOR_RGB2BGR))



