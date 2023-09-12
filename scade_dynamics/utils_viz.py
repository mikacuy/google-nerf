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

  skip = 15
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

  ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], marker=".", s=0.01, c="gray")
  ax.scatter(pc1[idx_chosen, 0], pc1[idx_chosen, 1], pc1[idx_chosen, 2], marker=".", s=10.0, c=colors1[idx_chosen])

  ax.set_xlim(-2,2)
  ax.set_ylim(-6,2)
  ax.set_zlim(-2,1)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  pc2 = pc2 + translation
  knn_pts = knn_pts + translation
  ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], marker=".", s=0.005, c="gray")
  ax.scatter(knn_pts[:, 0], knn_pts[:, 1], knn_pts[:, 2], marker=".", s=8.0, c=knn_pts_color)
  
  plt.savefig(fname) 


def save_pointcloud_samples(pcs, colors, fname, save_views=False):
  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(pcs[:, 0], pcs[:, 1], pcs[:, 2], marker=".", s=0.8, c=colors)
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

def save_pc_correspondences(pc1, pc2, colors1, colors2, fname, save_views=False):
  ### To visualize point clouds side by side
  translation = np.array([0.0, -4.0, 0.0])

  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], marker=".", s=0.4, c=colors1)
  ax.set_xlim(-2,2)
  ax.set_ylim(-6,2)
  ax.set_zlim(-2,1)
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  pc2 = pc2 + translation
  ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], marker=".", s=0.4, c=colors2)

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
