# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json
import cv2

sys.path.append('../')
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids

def read_file(rgb_file):
    fname = os.path.join(rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    return img

def read_cameras(pose_file, scene_path):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, 'r') as fp:
        meta = json.load(fp)

    near = float(meta['near'])
    far = float(meta['far'])
   

    rgb_files = []
    c2w_mats = []
    intrinsics = []
    
    for frame in meta['frames']:
        if len(frame['file_path']) != 0:
            # img, depth = read_files(scene_path, frame['file_path'], frame['depth_file_path'])
            rgb_files.append(os.path.join(scene_path, frame['file_path']))

        c2w = np.array(frame['transform_matrix'])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)

        fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
        intrinsics.append(get_intrinsics(fx, fy, cx, cy))


    c2w_mats = np.array(c2w_mats)
    intrinsics = np.array(intrinsics)

    return rgb_files, intrinsics, c2w_mats, near, far


def get_intrinsics_from_hwf(h, w, focal):
    return np.array([[focal, 0, 1.0*w/2, 0],
                     [0, focal, 1.0*h/2, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

def get_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx, 0],
                     [0, fy, cy, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

class ScannetDataset(Dataset):
    def __init__(self, args, mode,
                 # scenes=('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
                 scenes=(), **kwargs):
        self.folder_path = os.path.join("/orion/group/scannet_v2/dense_depth_priors/scenes/")
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == 'validation':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        self.mode = mode  # train / test / val

        scene = scenes[0]

        # if len(scenes) > 0:
        #     if isinstance(scenes, str):
        #         scenes = [scenes]
        # else:
        #     scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        self.render_rgb_files = []
        self.render_poses = []
        self.render_intrinsics = []


        self.scene_path = os.path.join(self.folder_path, scene)


        pose_file = os.path.join(self.scene_path, 'transforms_{}.json'.format(mode))
        rgb_files, intrinsics, poses, near, far = read_cameras(pose_file, self.scene_path)

        self.near = near
        self.far = far

        self.render_rgb_files.extend(rgb_files)
        self.render_poses.extend(poses)
        self.render_intrinsics.extend(intrinsics)

        ## Get number of images in the scene
        self.num_source_views = args.num_source_views


    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]

        train_pose_file = os.path.join(self.scene_path, 'transforms_train.json')
        train_rgb_files, train_intrinsics, train_poses, _, _ = read_cameras(train_pose_file, self.scene_path)


        if self.mode == 'train':
            id_render = idx
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        # rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        rgb = read_file(rgb_file)
        
        # rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), render_intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        # nearest_pose_ids = get_nearest_pose_ids(render_pose,
        #                                         train_poses,
        #                                         int(self.num_source_views*subsample_factor),
        #                                         tar_id=id_render,
        #                                         angular_dist_method='dist')
        # nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                1,
                                                tar_id=id_render,
                                                angular_dist_method='dist')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, 1, replace=False)


        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                              train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        near_depth = self.near
        far_depth = self.far

        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }

