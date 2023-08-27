import os

import numpy as np
import json
import cv2

import torchvision.transforms as transforms
import imageio
import torch
import torch.nn.functional as F

from skimage.transform import resize

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

# def load_ground_truth_depth(depth_file, depth_scaling_factor, near, far):
#     gt_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float64)[...,0]
#     gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)

#     return gt_depth

def load_ground_truth_depth(depth_file, depth_scaling_factor, near, far, downsample_scale=None):
    gt_depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float64)[...,0]
    gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)

    if downsample_scale is not None:
        gt_depth = cv2.resize(gt_depth, (int(gt_depth.shape[1]/downsample_scale), int(gt_depth.shape[0]/downsample_scale)), interpolation=cv2.INTER_NEAREST)

    return gt_depth

### With depth
def load_scene_blender_depth(basedir, half_res=True, train_skip=1, test_skip=5):

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    filenames = []

    json_fname =  os.path.join(basedir, 'transforms.json')

    with open(json_fname, 'r') as fp:
        meta = json.load(fp)

    near = 2.
    far = 6.
    camera_angle_x = float(meta['camera_angle_x'])

    skip = train_skip

    for frame in meta['frames'][::skip]:
        if len(frame['file_path']) != 0 :
            if half_res :
                downsample = 2
            else:
                downsample = 1

            img = read_files(os.path.join(basedir, frame['file_path'].split("/")[-1] +".png"), downsample_scale=downsample)

            max_depth = frame["max_depth"]
            depth_scaling_factor = (255. / max_depth)
            depth = load_ground_truth_depth(os.path.join(basedir, frame['depth_file_path'].split("/")[-1]+"0030.png"), depth_scaling_factor, near, far, downsample_scale=downsample)

            if depth.ndim == 2:
                depth = np.expand_dims(depth, -1)

            valid_depth = np.logical_and(depth[:, :, 0] > near, depth[:, :, 0] < far) # 0 values are invalid depth
            depth = np.clip(depth, near, far)

            filenames.append(frame['file_path'])
            all_imgs.append(img)
            all_depths.append(depth)
            all_valid_depths.append(valid_depth)

        # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
        all_poses.append(np.array(frame['transform_matrix']).astype(np.float32))

        H, W = img.shape[:2]
        focal = .5 * W / np.tan(.5 * camera_angle_x)                            

        fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
        all_intrinsics.append(np.array((fx, fy, cx, cy)).astype(np.float32))

    all_imgs = np.array(all_imgs)
    depths = np.array(all_depths)
    valid_depths = np.array(all_valid_depths)
    all_poses = np.array(all_poses)
    all_intrinsics = np.array(all_intrinsics)

    ### Split train and test
    i_test = np.arange(0, all_poses.shape[0], test_skip)
    i_train = np.setdiff1d(np.arange(all_poses.shape[0]), i_test)
    i_split = [i_train, i_test]

    ### For video -- use spherical poses
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    video_poses = []
    video_intrinsics = []
    for i in range(render_poses.shape[0]):
        video_poses.append(render_poses[i])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        focal /= downsample                               

        H, W = img.shape[:2]
        fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
        video_intrinsics.append(np.array((fx, fy, cx, cy)))

    video_poses = np.array(video_poses).astype(np.float32)
    video_intrinsics = np.array(video_intrinsics).astype(np.float32)

    print("=====Done loading data.=======")
    print(all_imgs.shape)
    print(all_poses.shape)
    print(depths.shape)
    print(valid_depths.shape)
    print(all_intrinsics.shape)
    print(video_poses.shape)
    print(video_intrinsics.shape)

    return all_imgs, depths, valid_depths, all_poses, H, W, all_intrinsics, near, far, i_split, video_poses, video_intrinsics, None

def read_feature(fname, feat_dim, H, W):
    curr_feat = torch.load(fname)

    ### This is for blender
    curr_feat = curr_feat.reshape((199, 199, feat_dim))

    curr_feat = curr_feat.permute(2, 0, 1).unsqueeze(0).float()
    curr_feat = F.interpolate(curr_feat, size=(H, W), mode='bilinear').squeeze().permute(1,2,0)

    return curr_feat

def load_scene_blender_depth_features(basedir, feature_dir, half_res=True, train_skip=1, test_skip=5, feat_dim=768):

    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    filenames = []

    all_features_fnames = []
    all_features = []

    count = 0

    json_fname =  os.path.join(basedir, 'transforms.json')

    with open(json_fname, 'r') as fp:
        meta = json.load(fp)

    near = 2.
    far = 6.
    camera_angle_x = float(meta['camera_angle_x'])

    skip = train_skip

    for frame in meta['frames'][::skip]:
        if len(frame['file_path']) != 0 :
            if half_res :
                downsample = 2
            else:
                downsample = 1

            img = read_files(os.path.join(basedir, frame['file_path'].split("/")[-1] +".png"), downsample_scale=downsample)

            curr_feat = torch.load(os.path.join(feature_dir, frame['file_path'].split("/")[-1] + '.pth')).cpu()
            # print(curr_feat.shape)

            if feat_dim == 768:
              curr_feat = curr_feat.reshape((199, 199, feat_dim))
              
            elif feat_dim == 384:
              curr_feat = curr_feat.reshape((99, 99, feat_dim))
              curr_feat = curr_feat.permute(2, 0, 1).unsqueeze(0).float()
              curr_feat = F.interpolate(curr_feat, size=(img.shape[0], img.shape[1]), mode='bilinear').squeeze().permute(1,2,0)
            # print(curr_feat.shape)

            all_features.append(curr_feat)

            curr_feat_fname = os.path.join(feature_dir, frame['file_path'].split("/")[-1] + '.pth')
            all_features_fnames.append(curr_feat_fname)

            print(count)
            count += 1

            max_depth = frame["max_depth"]
            depth_scaling_factor = (255. / max_depth)
            depth = load_ground_truth_depth(os.path.join(basedir, frame['depth_file_path'].split("/")[-1]+"0030.png"), depth_scaling_factor, near, far, downsample_scale=downsample)

            if depth.ndim == 2:
                depth = np.expand_dims(depth, -1)

            valid_depth = np.logical_and(depth[:, :, 0] > near, depth[:, :, 0] < far) # 0 values are invalid depth
            depth = np.clip(depth, near, far)

            filenames.append(frame['file_path'])
            all_imgs.append(img)
            all_depths.append(depth)
            all_valid_depths.append(valid_depth)

        # poses.append(np.array(frame['transform_matrix'])@ BLENDER2OPENCV)
        all_poses.append(np.array(frame['transform_matrix']).astype(np.float32))

        H, W = img.shape[:2]
        focal = .5 * W / np.tan(.5 * camera_angle_x)                            

        fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
        all_intrinsics.append(np.array((fx, fy, cx, cy)).astype(np.float32))

    all_imgs = np.array(all_imgs)
    depths = np.array(all_depths)
    valid_depths = np.array(all_valid_depths)
    all_poses = np.array(all_poses)
    all_intrinsics = np.array(all_intrinsics)

    all_features = torch.stack(all_features)
    # print(all_features.shape)
    # exit()

    ### Split train and test
    i_test = np.arange(0, all_poses.shape[0], test_skip)
    i_train = np.setdiff1d(np.arange(all_poses.shape[0]), i_test)
    i_split = [i_train, i_test]

    ### For video -- use spherical poses
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    video_poses = []
    video_intrinsics = []
    for i in range(render_poses.shape[0]):
        video_poses.append(render_poses[i])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
        focal /= downsample                               

        H, W = img.shape[:2]
        fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
        video_intrinsics.append(np.array((fx, fy, cx, cy)))

    video_poses = np.array(video_poses).astype(np.float32)
    video_intrinsics = np.array(video_intrinsics).astype(np.float32)

    print("=====Done loading data.=======")
    print(all_imgs.shape)
    print(all_poses.shape)
    print(depths.shape)
    print(valid_depths.shape)
    print(all_intrinsics.shape)
    print(video_poses.shape)
    print(video_intrinsics.shape)
    print(len(all_features_fnames))

    # return all_imgs, depths, valid_depths, all_poses, H, W, all_intrinsics, near, far, i_split, video_poses, video_intrinsics, None, all_features_fnames
    return all_imgs, depths, valid_depths, all_poses, H, W, all_intrinsics, near, far, i_split, video_poses, video_intrinsics, None, all_features, all_features_fnames
