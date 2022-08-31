import os
import numpy as np
import torch
from plyfile import PlyData, PlyElement


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
    pcd = pcd.astype(np.int)
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

def reconstruct_depth(depth, rgb, dir, pcd_name, focal):
    """
    para disp: disparity, [h, w]
    para rgb: rgb image, [h, w, 3], in rgb format
    """
    rgb = np.squeeze(rgb)
    depth = np.squeeze(depth)

    mask = depth < 1e-8
    depth[mask] = 0
    depth = depth / depth.max() * 10000

    pcd = reconstruct_3D(depth, f=focal)
    rgb_n = np.reshape(rgb, (-1, 3))
    save_point_cloud(pcd, rgb_n, os.path.join(dir, pcd_name + '.ply'))

def backup_files(log_dir, train_fname):
    ### For training file backups
    os.system('cp %s %s' % (train_fname, log_dir)) # bkp of model def
    os.system('cp -r lib/ %s' % (log_dir)) # bkp of train procedure
    os.system('cp -r data/ %s' % (log_dir)) # bkp of data utils
    os.system('cp %s %s' % ("tools/parse_arg_base.py", log_dir)) # bkp of model def
    os.system('cp %s %s' % ("tools/parse_arg_train.py", log_dir)) # bkp of model def
    os.system('cp %s %s' % ("tools/parse_arg_val.py", log_dir)) # bkp of model def
    os.system('cp %s %s' % ("tools/utils.py", log_dir)) # bkp of model def



def load_mean_var_adain(fname, device):
    input_dict = np.load(fname, allow_pickle=True)

    mean0 = input_dict.item().get('mean0')
    mean1 = input_dict.item().get('mean1')
    mean2 = input_dict.item().get('mean2')
    mean3 = input_dict.item().get('mean3')

    var0 = input_dict.item().get('var0')
    var1 = input_dict.item().get('var1')
    var2 = input_dict.item().get('var2')
    var3 = input_dict.item().get('var3')

    mean0 = torch.from_numpy(mean0).to(device=device)
    mean1 = torch.from_numpy(mean1).to(device=device)
    mean2 = torch.from_numpy(mean2).to(device=device)
    mean3 = torch.from_numpy(mean3).to(device=device)
    var0 = torch.from_numpy(var0).to(device=device)
    var1 = torch.from_numpy(var1).to(device=device)
    var2 = torch.from_numpy(var2).to(device=device)
    var3 = torch.from_numpy(var3).to(device=device)

    return mean0, var0, mean1, var1, mean2, var2, mean3, var3























    