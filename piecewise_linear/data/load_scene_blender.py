import os

import numpy as np
import json
import cv2

import torchvision.transforms as transforms
import imageio
import torch

#### For Pose Corrections ####
BLENDER2OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    # c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).cpu().numpy() @ c2w.cpu().numpy() @ BLENDER2OPENCV
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])).cpu().numpy() @ c2w.cpu().numpy()
    return c2w


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

def load_scene_blender(basedir, train_json = "transforms_train.json", half_res=True):
    splits = ['train', 'val', 'test', 'video']

    all_imgs = []

    all_poses = []
    all_intrinsics = []
    counts = [0]
    filenames = []
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):

            json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

            with open(json_fname, 'r') as fp:
                meta = json.load(fp)

            if 'train' in s:
                near = 2.
                far = 6.
                camera_angle_x = float(meta['camera_angle_x'])

            imgs = []
            poses = []
            intrinsics = []

            if s=='train':
                skip = 1
            else:
                skip = 8
            
            for frame in meta['frames'][::skip]:
                if len(frame['file_path']) != 0 :
                    if half_res :
                        downsample = 2
                    else:
                        downsample = 1

                    img = read_files(os.path.join(basedir, frame['file_path']+".png"), downsample_scale=downsample)

                    filenames.append(frame['file_path'])
                    imgs.append(img)

                # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
                poses.append(np.array(frame['transform_matrix']))

                H, W = img.shape[:2]
                focal = .5 * W / np.tan(.5 * camera_angle_x)                            

                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            if len(imgs) > 0:
                all_imgs.append(np.array(imgs))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        elif s == "video":
            ### Use spherical poses
            render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

            poses = []
            intrinsics = []
            for i in range(render_poses.shape[0]):
                poses.append(render_poses[i])
                focal = .5 * W / np.tan(.5 * camera_angle_x)
                focal /= downsample                               

                H, W = img.shape[:2]
                fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
                intrinsics.append(np.array((fx, fy, cx, cy)))

            counts.append(counts[-1] + len(poses))
            all_poses.append(np.array(poses).astype(np.float32))
            all_intrinsics.append(np.array(intrinsics).astype(np.float32))

        else:
            counts.append(counts[-1])

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    
    return imgs, None, None, poses, H, W, intrinsics, near, far, i_split, None, None




























