import sys
import os
import numpy as np
import json
import cv2
import torchvision.transforms as transforms
import imageio
import torch
import argparse
import os.path as osp
from tqdm import tqdm


parser = argparse.ArgumentParser(description='NeRF data post processing.')
parser.add_argument('img_dir', type=str, help='Image directory')
parser.add_argument('--out_dir', type=str,
                    default=None, help='Image directory')
args = parser.parse_args()


def to8b(x):
    # print(x.dtype, x.max(), x.min(), x.mean())
    return (255*np.clip(x,0,1)).astype(np.uint8)

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

folder = args.img_dir
# "../data/nerf_synthetic_multicam2/lego_0.5_1.25/"
out_folder = args.out_dir
if out_folder is None:
    out_folder = folder
    if out_folder.endswith("/"):
        out_folder = out_folder[:-1]
    out_folder = out_folder + "_rgb"
# "../data/nerf_synthetic_multicam2/lego_0.5_1.25_rgb/"


flst = list(os.listdir(folder))
# for f in os.listdir(folder):
for f in tqdm(flst):
    if not f.endswith(".png"): continue
    fname = osp.join(folder, f)

    img = read_files(fname)
    if img.shape[-1] > 3:
        img[..., :3] *= img[..., 3:]
    img = img[..., :3]
    img_out = cv2.cvtColor(to8b(img), cv2.COLOR_RGB2BGR)
    out_fname = osp.join(out_folder, f)
    status = cv2.imwrite(out_fname, img_out)
    print(out_fname, status)
