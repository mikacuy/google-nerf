import os

import numpy as np
import json
import cv2

import torchvision.transforms as transforms
import imageio
import torch

import glob

#### For Pose Corrections ####
BLENDER2OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    # 草，这个地方源代码没有乘这个blender2opencv，做这个操作相当于把相机转换到另一个坐标系了，和一般的nerf坐标系不同
    poses_centered = poses_centered @ BLENDER2OPENCV
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)
    print('center in center_poses',poses_centered[:, :3, 3].mean(0))

    return poses_centered, np.linalg.inv(pose_avg_homo) @ BLENDER2OPENCV
######################################

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


def load_scene_llff(basedir, downsample=6.):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:

        if s in ["train", "test"]:
            poses_bounds = np.load(os.path.join(basedir, 'poses_bounds.npy'))  # (N_images, 17)
            image_paths = sorted(glob.glob(os.path.join(basedir, 'images/*')))

            print(len(poses_bounds) , len(image_paths), basedir)
            assert len(poses_bounds) == len(image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

            poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
            bounds = poses_bounds[:, -2:]  # (N_images, 2)

            # Step 1: rescale focal length according to training resolution
            H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images

            ### Need to set this up if we want to resize
            focal = focal/downsample
            H = int(H/downsample)
            W = int(W/downsample)

            # Step 2: correct poses
            # Original poses has rotation in form "down right back", change to "right up back"
            # See https://github.com/bmild/nerf/issues/34
            poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
            # (N_images, 3, 4) exclude H, W, focal
            poses, pose_avg = center_poses(poses)
            # print('pose_avg in read_meta', self.pose_avg)
            # self.poses = poses @ self.blender2opencv

            # Step 3: correct scale so that the nearest depth is at a little more than 1.0
            # See https://github.com/bmild/nerf/issues/34
            near_original = bounds.min()
            scale_factor = near_original * 0.75  # 0.75 is the default parameter
            print('scale_factor', scale_factor)
            # the nearest depth is at 1/0.75=1.33
            bounds /= scale_factor
            poses[..., 3] /= scale_factor

            near = np.min(bounds[..., 0])*0.8
            far = np.max(bounds[..., 1])*1.2  # focus on central object only

            # sub select training views from pairing file
            if os.path.exists(os.path.join(basedir, "..", 'pairs.th')):
                name = os.path.basename(basedir.split("/")[-1])
                img_idx = torch.load(os.path.join(basedir, "..", 'pairs.th'))[f'{name}_{s}']

                print(f'===> {s}ing index: {img_idx}')
               
            imgs = []
            curr_poses = []
            intrinsics = []
            
            for i in img_idx:

                image_path = image_paths[i]
                c2w = torch.FloatTensor(poses[i])

                img = read_files(image_paths[i], downsample_scale=downsample)
                imgs.append(img)
                curr_poses.append(poses[i])
                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

                # img = Image.open(image_path).convert('RGB')
                # img = img.resize(self.img_wh, Image.LANCZOS)
                # img = self.transform(img)  # (3, h, w)

            counts.append(counts[-1] + len(curr_poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
            all_poses.append(np.array(curr_poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)


    return imgs, None, None, poses, H, W, intrinsics, near, far, i_split, None, None



