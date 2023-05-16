import torch
import glob
import numpy as np
import os
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from .ray_utils import get_ray_directions

from .base import BaseDataset


class NSVFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        xyz_min, xyz_max = \
            np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
        self.shift = (xyz_max+xyz_min)/2
        self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

        if 'Synthetic' in root_dir or 'Ignatius' in root_dir:
            # hard-code fix the bound error for some scenes...
            if 'Mic' in root_dir: self.scale *= 1.2
            elif 'Lego' in root_dir: self.scale *= 1.1
            ###################################################
            with open(os.path.join(root_dir, 'intrinsics.txt')) as f:
                fx = fy = float(f.readline().split()[0]) * downsample
            if 'Synthetic' in root_dir:
                w = h = int(800*downsample)
            else:
                w, h = int(1920*downsample), int(1080*downsample)

            K = np.float32([[fx, 0, w/2],
                            [0, fy, h/2],
                            [0,  0,   1]])
        else:
            K = np.loadtxt(os.path.join(root_dir, 'intrinsics.txt'),
                           dtype=np.float32)[:3, :3]
            if 'BlendedMVS' in root_dir:
                w, h = int(768*downsample), int(576*downsample)
            elif 'Tanks' in root_dir:
                w, h = int(1920*downsample), int(1080*downsample)
            K[:2] *= downsample
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

        # print(self.K)
        # print(self.directions)
        # exit()

        self.read_meta(split)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'test_traj': # BlendedMVS and TanksAndTemple
            if 'Ignatius' in self.root_dir:
                poses_path = \
                    sorted(glob.glob(os.path.join(self.root_dir, 'test_pose/*.txt')))
                poses = [np.loadtxt(p) for p in poses_path]
            else:
                poses = np.loadtxt(os.path.join(self.root_dir, 'test_traj.txt'))
                poses = poses.reshape(-1, 4, 4)
            for pose in poses:
                c2w = pose[:3]
                c2w[:, 0] *= -1 # [left down front] to [right down front]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]
        else:
            if split == 'train': prefix = '0_'
            elif split == 'trainval': prefix = '[0-1]_'
            elif 'Synthetic' in self.root_dir: prefix = '2_'
            elif split == 'test': prefix = '1_' # test set for real scenes
            else: raise ValueError(f'{split} split not recognized!')
            imgs = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix+'*.png')))
            poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix+'*.txt')))

            print(f'Loading {len(imgs)} {split} images ...')
            for img, pose in tqdm(zip(imgs, poses)):
                c2w = np.loadtxt(pose)[:3]
                c2w[:, 3] -= self.shift
                c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]
                self.poses += [c2w]

                img = Image.open(img)

                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (c, h, w)
                img = rearrange(img, 'c h w -> (h w) c')

                if 'Jade' in self.root_dir or 'Fountain' in self.root_dir:
                    # these scenes have black background, changing to white
                    img[torch.all(img<=0.1, dim=-1)] = 1.0
                if img.shape[-1] == 4:
                    img = img[:, :3]*img[:, -1:]+(1-img[:, -1:]) # blend A to RGB

                self.rays += [img]

            self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)






