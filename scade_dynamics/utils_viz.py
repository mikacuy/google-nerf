### For visualization
import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


def save_pointcloud_samples(pcs, colors, fname):
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

def save_motion_vectors(pcs, colors, vecs, fname):
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
  endpoint_2 = pcs - vecs
  line_to_draw = np.array([endpoint_1, endpoint_2])

  for pt_idx in range(line_to_draw.shape[1]):
    ax.plot3D(line_to_draw[:, pt_idx, 0], line_to_draw[:, pt_idx, 1], line_to_draw[:, pt_idx, 2], color= "blue", linewidth=0.2)

  plt.savefig(fname)