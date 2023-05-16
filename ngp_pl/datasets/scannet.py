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

    fx = 577.870605
    fy = 577.870605
    cx = 319.500000
    cy = 239.500000

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 8)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx+0.5) / fx, 0)
    y = np.where(valid, z * (r - cy+0.5) / fy, 0)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    return np.dstack((x, y, z))

class ScannetDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

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

        self.shift = (xyz_max+xyz_min)/2
        self.scale = (xyz_max-xyz_min).max()/2 * 1.05 # enlarge a little

        print("Scene scale:")
        print(self.scale)
        print()
        # exit()

        # K = np.loadtxt(os.path.join(root_dir, 'intrinsic_color.txt'),
        #                dtype=np.float32)[:3, :3]


        ### This intrinsics seems to be more correct
        K = np.loadtxt(os.path.join(root_dir, 'intrinsic_depth.txt'),
                       dtype=np.float32)[:3, :3]

        print("Downsample: "+str(downsample))
        w, h = int(640*downsample), int(480*downsample)
        print("Image size: "+str(w)+" "+str(h))

        # K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # K[:2] *= downsample
        # K[:2] /= 2.

        self.K = torch.FloatTensor(K)

        # self.directions = get_ray_directions(h, w, self.K)

        self.directions = get_ray_directions_scannet(h, w, self.K)  ## not flipped
        # self.directions = get_ray_directions_scannet_v2(h, w, self.K) ##flipped y and z

        self.img_wh = (w, h)

        ### Train/test split (test frames are every test_skip frames in the trajectory)
        self.test_skip = kwargs["test_skip"]

        self.read_meta(split, rot_transpose=kwargs["rot_transpose"], scale_flip=kwargs["scale_flip"])

    def read_meta(self, split, rot_transpose=False, scale_flip=False):
        self.rays = []
        self.poses = []

        filename = os.path.join(self.root_dir, "test_step_"+str(self.test_skip), split+".txt")
        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]

        print(f'Loading {len(lines)} {split} images ...')
        print("Rot transpose: "+str(rot_transpose))
        print("Scale flip: "+str(scale_flip))

        for line in lines:
            ### Each example
            pose = os.path.join(self.root_dir, "pose", line+".txt")

            with open(pose, "r") as f:
                inner_lines = f.readlines()
            ls = []
            for inner_line in inner_lines:
                l = list(map(float, inner_line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)

            c2w = c2w[:3]

            ### Scannet alignment does not require this
            if rot_transpose:
                c2w[:,:3] = c2w[:,:3].T
            if scale_flip:
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1 

            c2w[:, 3] -= self.shift
            c2w[:, 3] /= 2*self.scale # to bound the scene inside [-0.5, 0.5]

            # ###########################################
            # ############### DEBUG ####################
            # ###########################################

            # ### Transform depth map and align with whole scene
            # ### Works without modifying c2W (no transpose no flipped axis)

            # ### First debug visualize depth map
            # print(line)
            # depth_fname = os.path.join(self.root_dir, "depth", line+".png")
            # img = np.array(imageio.imread(depth_fname))
            # img = img.astype(float)/1000.

            # vmax = np.percentile(img, 95)
            # normalizer = pyplot.colors.Normalize(vmin=img.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
            # colormapped_im = (mapper.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
            # im = Image.fromarray(colormapped_im)
            # im.save('test.png')

            # ### Export to point cloud           
            # pc = point_cloud_from_depth(img).reshape(-1, 3)
            # print(pc.shape)

            # ### Transform point cloud
            # print(c2w.shape)
            # # c2w = np.vstack((c2w, np.array([0,0,0,1])))
            
            # R = c2w[:,:3]
            # t = c2w[:,3]

            # pc =  pc @ R.T + t 

            # print(pc.shape)

            # cloud=trimesh.PointCloud(pc)
            # cloud.export('unprojected_depth_ss.ply')


            # ### world coor
            # raw_scan_rootdir = "/orion/group/scannet_v2/scans/"
            # scenename = self.root_dir.split("/")[-1]            
            # with open(os.path.join(raw_scan_rootdir, scenename, scenename+"_vh_clean_2.ply"), 'rb') as f:
            #     plydata = PlyData.read(f)
            #     num_verts = plydata['vertex'].count
            #     pts3d = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            #     pts3d[:,0] = plydata['vertex'].data['x']
            #     pts3d[:,1] = plydata['vertex'].data['y']
            #     pts3d[:,2] = plydata['vertex'].data['z']

            #     pts3d -= self.shift
            #     # pts3d = 2*self.scale

            #     print(pts3d.shape)
            #     cloud=trimesh.PointCloud(pts3d)
            #     cloud.export('whole_scene_ss.ply')


            # #### Plot camera origin and directions
            # print("Debugging camera rays...")
            # print(self.directions.shape)
            # c2w_feed = torch.from_numpy(c2w).float()

            # print(c2w_feed.shape)

            # rays_o, rays_d = get_rays(self.directions, c2w_feed)

            # rays_o = rays_o.cpu().numpy()
            # rays_d = rays_d.cpu().numpy()

            # ### Debug camera rays
            # step = 0.25
            # iterations = 10
            # pts = []
            # for i in range(iterations):
            #     curr_pts = rays_o+step*(i)*rays_d
            #     # curr_pts = rays_o + step*(i)*rays_d*(-1.)

            #     print(curr_pts.shape)
            #     pts.append(curr_pts)

            # pts = np.array(pts)
            # print(pts.shape)
            # pts = pts.reshape(-1, 3)
            # print(pts.shape)
            # cloud=trimesh.PointCloud(pts)
            # cloud.export('camera_debug_not_flipped.ply')
            # exit()

            # ###########################################            
            # ###########################################
            # ###########################################


            self.poses += [c2w]

            img = os.path.join(self.root_dir, "rgb", line+".jpg")
            img = imageio.imread(img).astype(np.float32)/255.0
            img = cv2.resize(img, self.img_wh)

            img = self.transform(img) # (c, h, w)

            img = rearrange(img, 'c h w -> (h w) c')

            self.rays += [img]


        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)






