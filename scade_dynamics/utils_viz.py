### For visualization
import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from plyfile import PlyData, PlyElement
import os, sys

import matplotlib as mpl
import matplotlib.cm as cm

def save_pointcloud_noise(pcs, noise, fname, size=0.8):
  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
  cmap = cm.viridis
  m = cm.ScalarMappable(norm=norm, cmap=cmap)

  ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], marker=".", s=size, c=m.to_rgba(noise))
  ax.set_xlim(-2,2)
  ax.set_ylim(-2,2)
  ax.set_zlim(-2,1)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')
  plt.savefig(fname)

def save_pc_correspondences_samples_iteration(pc1, pc2, colors1, noise, samples, samples_colors, selected_neighbors, selected_neighbors_noise, fname):
  ### To visualize point clouds side by side
  translation = np.array([0.0, -4.0, 0.0])

  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], marker=".", s=0.005, c="gray")
  ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], marker=".", s=3.0, c=samples_colors)

  
  ax.set_xlim(-2,2)
  ax.set_ylim(-6,2)
  ax.set_zlim(-2,1)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
  cmap = cm.viridis
  m = cm.ScalarMappable(norm=norm, cmap=cmap)

  pc2 = pc2 + translation
  ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], marker=".", s=0.01, c=m.to_rgba(noise))

  endpoint_1 = samples
  endpoint_2 = selected_neighbors + translation
  line_to_draw = np.array([endpoint_1, endpoint_2])

  ax.scatter(endpoint_2[:, 0], endpoint_2[:, 1], endpoint_2[:, 2], marker="x", s=1.0, c=m.to_rgba(selected_neighbors_noise))

  # skip = 15
  skip = 50
  for pt_idx in range(line_to_draw.shape[1]):
    if pt_idx % skip != 0 :
      continue  
    # ax.plot3D(line_to_draw[:, pt_idx, 0], line_to_draw[:, pt_idx, 1], line_to_draw[:, pt_idx, 2], color= "blue", linewidth=0.05)
    ax.plot3D(line_to_draw[:, pt_idx, 0], line_to_draw[:, pt_idx, 1], line_to_draw[:, pt_idx, 2], color= samples_colors[pt_idx], linewidth=0.8)
  
  plt.savefig(fname) 

def save_pc_knn_samples(pc1, colors1, idx_chosen, pc2, knn_pts, knn_pts_color, fname):
  ### To visualize point clouds side by side
  translation = np.array([0.0, -4.0, 0.0])

  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], marker=".", s=0.05, c="gray")
  ax.scatter(pc1[idx_chosen, 0], pc1[idx_chosen, 1], pc1[idx_chosen, 2], marker=".", s=20.0, c=colors1[idx_chosen])

  ### this coordinate is for chair
  ax.set_xlim(-1.5,2.5)
  ax.set_ylim(-5,4)
  ax.set_zlim(-2,1)
  
  ### this was for hotdog
  # ax.set_xlim(-2,2)
  # ax.set_ylim(-6,2)
  # ax.set_zlim(-2,1)

  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  pc2 = pc2 + translation
  knn_pts = knn_pts + translation
  ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], marker=".", s=0.005, c="gray")
  ax.scatter(knn_pts[:, 0], knn_pts[:, 1], knn_pts[:, 2], marker=".", s=8.0, c=knn_pts_color)
  
  ### Used for the chair scene
  ax.view_init(elev=40., azim=230.)

  plt.savefig(fname) 


def save_pointcloud_samples(pcs, colors, fname, save_views=False, pc_size=0.8, skip=1):
  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(pcs[::skip, 0], pcs[::skip, 1], pcs[::skip, 2], marker=".", s=pc_size, c=colors[::skip])
  ax.set_xlim(-2,2)
  ax.set_ylim(-2,2)
  ax.set_zlim(-2,1)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')
  plt.savefig(fname)

  if save_views:
    fname = fname[:-4]
    vid_dir = os.path.join(fname + "_video")
    os.makedirs(vid_dir, exist_ok=True)

    for ii in range(0,360,30):
        ax.view_init(elev=45., azim=ii)
        plt.savefig(os.path.join(vid_dir, str(ii) + ".png"))


def save_motion_vectors(pcs, colors, vecs, fname, is_vec=True):
  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], marker=".", s=0.4, c=colors)
  ax.set_xlim(-2,2)
  ax.set_ylim(-2,2)
  ax.set_zlim(-2,1)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  endpoint_1 = pcs

  if is_vec:
    endpoint_2 = pcs - vecs
  else:
    endpoint_2 = - vecs
  line_to_draw = np.array([endpoint_1, endpoint_2])

  skip = 100
  
  for pt_idx in range(line_to_draw.shape[1]):
    if pt_idx % skip != 0 :
      continue
    ax.plot3D(line_to_draw[:, pt_idx, 0], line_to_draw[:, pt_idx, 1], line_to_draw[:, pt_idx, 2], color= colors[pt_idx], linewidth=0.8)

  plt.savefig(fname)

def save_pc_correspondences(pc1, pc2, colors1, colors2, fname, save_views=False, size=0.4):
  ### To visualize point clouds side by side
  translation = np.array([0.0, -4.0, 0.0])

  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], marker=".", s=size, c=colors1)
  ax.set_xlim(-2,2)
  ax.set_ylim(-6,2)
  ax.set_zlim(-2,1)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  pc2 = pc2 + translation
  ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], marker=".", s=size, c=colors2)

  endpoint_1 = pc1
  endpoint_2 = pc2
  line_to_draw = np.array([endpoint_1, endpoint_2])

  skip = 200
  
  for pt_idx in range(line_to_draw.shape[1]):
    if pt_idx % skip != 0 :
      continue

    # ax.plot3D(line_to_draw[:, pt_idx, 0], line_to_draw[:, pt_idx, 1], line_to_draw[:, pt_idx, 2], color= "blue", linewidth=0.05)
    ax.plot3D(line_to_draw[:, pt_idx, 0], line_to_draw[:, pt_idx, 1], line_to_draw[:, pt_idx, 2], color= colors1[pt_idx], linewidth=0.8)
  
  plt.savefig(fname)

  if save_views:
    fname = fname[:-4]
    vid_dir = os.path.join(fname + "_video")
    os.makedirs(vid_dir, exist_ok=True)

    for ii in range(0,360,30):
        ax.view_init(elev=45., azim=ii)
        plt.savefig(os.path.join(vid_dir, str(ii) + ".png"))



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


### For 2D optical flow
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)