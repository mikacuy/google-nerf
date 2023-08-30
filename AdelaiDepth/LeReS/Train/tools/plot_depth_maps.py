import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import json
import cv2

def reconstruct_3D(depth, f):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    cu = depth.shape[1] / 2
    cv = depth.shape[0] / 2
    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    if f > 1e5:
        print('Infinite focal length!!!')
        x = u - cu
        y = v - cv
        z = depth / depth.max() * x.max()
    else:
        x = (u - cu) * depth / f
        y = (v - cv) * depth / f
        z = depth

    x = np.reshape(x, (width * height, 1)).astype(np.float)
    y = np.reshape(y, (width * height, 1)).astype(np.float)
    z = np.reshape(z, (width * height, 1)).astype(np.float)
    pcd = np.concatenate((x, y, z), axis=1)
    # pcd = pcd.astype(np.int)
    return pcd

def reconstruct_3D_intrinsics(depth, intrinsic):
    """
    Reconstruct depth to 3D pointcloud with the provided focal length.
    Return:
        pcd: N X 3 array, point cloud
    """
    fx, fy, cu, cv = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]

    # cu = depth.shape[1] / 2
    # cv = depth.shape[0] / 2

    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for i in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for i in np.arange(width)])
    v = v.transpose(1, 0)

    x = (u - cu) * depth / fx
    y = (v - cv) * depth / fy
    z = depth

    x = np.reshape(x, (width * height, 1)).astype(np.float)
    y = np.reshape(y, (width * height, 1)).astype(np.float)
    z = np.reshape(z, (width * height, 1)).astype(np.float)
    pcd = np.concatenate((x, y, z), axis=1)
    # pcd = pcd.astype(np.int)
    return pcd

def save_point_cloud(pcd, rgb, filename, binary=True):
    """Save an RGB point cloud as a PLY file.

    :paras
      @pcd: Nx3 matrix, the XYZ coordinates
      @rgb: NX3 matrix, the rgb colors for each 3D point
    """
    assert pcd.shape[0] == rgb.shape[0]

    if rgb is None:
        gray_concat = np.tile(np.array([128], dtype=np.uint8), (pcd.shape[0], 3))
        points_3d = np.hstack((pcd, gray_concat))
    else:
        points_3d = np.hstack((pcd, rgb))
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

         # Write
        PlyData([el]).write(filename)
    else:
        x = np.squeeze(points_3d[:, 0])
        y = np.squeeze(points_3d[:, 1])
        z = np.squeeze(points_3d[:, 2])
        r = np.squeeze(points_3d[:, 3])
        g = np.squeeze(points_3d[:, 4])
        b = np.squeeze(points_3d[:, 5])

        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % r.shape[0]
        # ---- Save ply data to disk
        np.savetxt(filename, np.column_stack((x, y, z, r, g, b)), fmt="%d %d %d %d %d %d", header=ply_head, comments='')

def reconstruct_depth(depth, rgb, dir, pcd_name, focal, scale=1.0):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    rgb = np.squeeze(rgb)
    depth = np.squeeze(depth)

    mask = depth < 1e-8
    depth[mask] = 0

    # print(depth.max())
    # exit()
    # depth = depth / depth.max() * scale
    depth = depth * scale

    pcd = reconstruct_3D(depth, f=focal)
    rgb_n = np.reshape(rgb, (-1, 3))
    save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))

def reconstruct_depth_intrinsics(depth, rgb, dir, pcd_name, intrinsic, scale=1.0):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    rgb = np.squeeze(rgb)
    depth = np.squeeze(depth)

    mask = depth < 1e-8
    depth[mask] = 0

    # print(depth.max())
    # exit()
    # depth = depth / depth.max() * scale
    depth = depth * scale

    pcd = reconstruct_3D_intrinsics(depth, intrinsic)
    rgb_n = np.reshape(rgb, (-1, 3))
    save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))

def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return img, depth

DATA_ROOT = "/orion/group/scannet_v2/dense_depth_priors/scenes" 
SCENE_ID = "scene0710_00"
basedir = os.path.join(DATA_ROOT, SCENE_ID)

DUMP_DIR = "dump_recons_710"
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

if os.path.exists(os.path.join(basedir, 'transforms_train.json')):

    json_fname =  os.path.join(basedir, 'transforms_train.json')

    with open(json_fname, 'r') as fp:
        meta = json.load(fp)

    near = float(meta['near'])
    far = float(meta['far'])
    depth_scaling_factor = float(meta['depth_scaling_factor'])
   
    
    for frame in meta['frames']:
        if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
            img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])
            
            if depth.ndim == 2:
                depth = np.expand_dims(depth, -1)

            valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
            depth = (depth / depth_scaling_factor).astype(np.float32)

        poses.append(np.array(frame['transform_matrix']))
        H, W = img.shape[:2]
        fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
        intrinsics = np.array((fx, fy, cx, cy))

        print(img)
        print(depth)
        print(depth.shape)
        exit()











