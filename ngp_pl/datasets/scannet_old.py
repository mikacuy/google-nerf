import torch
import glob
import numpy as np
import os
from PIL import Image
from einops import rearrange
from tqdm import tqdm

import cv2
import matplotlib as pyplot
import matplotlib.cm as cm

from .ray_utils import *

from .base import BaseDataset
import trimesh
import imageio
from plyfile import PlyData, PlyElement

def point_cloud_from_depth(depth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """

    # fx = 577.870605
    # fy = 577.870605
    # cx = 0.
    # cy = 0.

    fx = 577.870605/2.
    fy = 577.870605/2.
    cx = 319.500000/2.
    cy = 239.500000/2.

    # fx = 1170.187988
    # fy = 1170.187988
    # cx = 647.750000
    # cy = 483.750000

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth / 255.0, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))

class ScannetDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        # xyz_min, xyz_max = \
        #     np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)

        # print(xyz_min)
        # print(xyz_max)

        raw_scan_rootdir = "/orion/group/scannet_v2/scans/"
        scenename = root_dir.split("/")[-1]

        with open(os.path.join(raw_scan_rootdir, scenename, scenename+"_vh_clean_2.ply"), 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            vertices[:,0] = plydata['vertex'].data['x']
            vertices[:,1] = plydata['vertex'].data['y']
            vertices[:,2] = plydata['vertex'].data['z']

            x_min = np.min(vertices[:,0])
            y_min = np.min(vertices[:,1])
            z_min = np.min(vertices[:,2])

            x_max = np.max(vertices[:,0])
            y_max = np.max(vertices[:,1])
            z_max = np.max(vertices[:,2])

            xyz_min = np.array([x_min, y_min, z_min])
            xyz_max = np.array([x_max, y_max, z_max])

        # print(vertices.shape)
        # cloud=trimesh.PointCloud(vertices)
        # cloud.export('scene.ply')


        self.shift = (xyz_max+xyz_min)/2
        self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little


        K = np.loadtxt(os.path.join(root_dir, 'intrinsic_color.txt'),
                       dtype=np.float32)[:3, :3]

        w, h = int(640*downsample), int(480*downsample)

        # K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # K[:2] *= downsample
        # K[:2] /= 2.


        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

        # print(self.img_wh)
        # print(self.K)
        # print(self.directions.shape)
        # exit()

        ### Train/test split (test frames are every test_skip frames in the trajectory)
        self.test_skip = kwargs["test_skip"]

        self.read_meta(split)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        filename = os.path.join(self.root_dir, "test_step_"+str(self.test_skip), split+".txt")
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]

        print(f'Loading {len(lines)} {split} images ...')
        for line in lines:
            ### Each example
            pose = os.path.join(self.root_dir, "pose", line+".txt")
            img = os.path.join(self.root_dir, "rgb", line+".jpg")


            ################ DEBUG ####################
            # ### First debug visualize depth map
            # print(line)
            # depth_fname = os.path.join(self.root_dir, "depth", line+".png")
            # img = np.array(imageio.imread(depth_fname))
            # print(np.max(img))
            # img = img.astype(float)/np.max(img) * 255.
            # print(img)

            # vmax = np.percentile(img, 95)
            # normalizer = pyplot.colors.Normalize(vmin=img.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
            # colormapped_im = (mapper.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
            # im = Image.fromarray(colormapped_im)
            # im.save('test.png')

            # ### Export to point cloud           
            # pc = point_cloud_from_depth(img).reshape(-1, 3)
            # print(pc.shape)
            # cloud=trimesh.PointCloud(pc)
            # cloud.export('sample.ply')

            # c2w = np.loadtxt(pose)
            # print(c2w)
            # exit()
            ############################################

            with open(pose, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = c2w[:3]         

            # ### Old
            # c2w = np.loadtxt(pose)[:3]

            # ##### Debug: try to use inverse
            # bottom = np.array([[0., 0., 0., 1.]])
            # c2w = np.concatenate([c2w, bottom], 0)
            # c2w = np.linalg.inv(c2w)[:3]
            # ##################

            c2w[:, 3] -= self.shift
            c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]

            self.poses += [c2w]

            img = Image.open(img)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (c, h, w)
            img = rearrange(img, 'c h w -> (h w) c')

            self.rays += [img]


        # # # ### Mika TODO center poses and scale
        # # # ### FIX ---> !!!
        # raw_scan_rootdir = "/orion/group/scannet_v2/scans/"
        # scenename = self.root_dir.split("/")[-1]

        # with open(os.path.join(raw_scan_rootdir, scenename, scenename+"_vh_clean_2.ply"), 'rb') as f:
        #     plydata = PlyData.read(f)
        #     num_verts = plydata['vertex'].count
        #     pts3d = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        #     pts3d[:,0] = plydata['vertex'].data['x']
        #     pts3d[:,1] = plydata['vertex'].data['y']
        #     pts3d[:,2] = plydata['vertex'].data['z']

        # self.poses = np.stack(self.poses, axis=0)
        # self.poses, pts3d = center_poses(self.poses, pts3d)
        # scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        # self.poses[..., 3] /= 2*scale


        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)






