"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion

import argparse
import os, sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/scenes/scene0758_00/', help='Root dir for dataset')
# parser.add_argument('--logdir', default='/orion/u/mikacuy/coordinate_mvs/dense_depth_priors_nerf/log_scene758_videodemo_finetune/scene0758_freezess_lr5e4_clr5e6/', help='Root dir for dataset')
parser.add_argument('--logdir', default='/orion/u/mikacuy/coordinate_mvs/dense_depth_priors_nerf/Scannet/scene758/scene0758_00_ddp_outdomain/', help='Root dir for dataset')
# parser.add_argument('--logdir', default='/orion/u/mikacuy/coordinate_mvs/dense_depth_priors_nerf/log_scene0758_00/20220826_152210_scene0758_00/', help='Root dir for dataset')
FLAGS = parser.parse_args()


blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def get_intrinsics(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

if __name__ == "__main__":

  json_fname =  os.path.join(FLAGS.dataroot, 'transforms_video.json')
  with open(json_fname, 'r') as fp:
      meta = json.load(fp)

  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #

  print("Estimating voxel volume bounds...")
  start_idx = 0
  end_idx = len(meta['frames'])
  vol_bnds = np.zeros((3,2))

  for img_idx in range(start_idx, end_idx, 12):
    frame = meta['frames'][img_idx]
    cam_pose = np.array(frame['transform_matrix']) @ blender2opencv
    fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
    cam_intr = get_intrinsics(fx, fy, cx, cy)

    # depth_fname = os.path.join(FLAGS.logdir, "video_demo2_depth_demo", str(img_idx)+".png")
    depth_fname = os.path.join(FLAGS.logdir, "video_demo2_depth_demo2", str(img_idx)+".png")
    depth_im = cv2.imread(depth_fname,-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)    

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))



  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)
  # tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.1)
  # tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.01)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for img_idx in range(start_idx, end_idx, 12):
    frame = meta['frames'][img_idx]
    print("Fusing frame %d/%d"%(img_idx, end_idx))

    # Read RGB-D image and camera pose

    # depth_fname = os.path.join(FLAGS.logdir, "video_demo2_depth_demo", str(img_idx)+".png")
    # img_fname = os.path.join(FLAGS.logdir, "video_demo2_demo", str(img_idx)+".jpg")

    depth_fname = os.path.join(FLAGS.logdir, "video_demo2_depth_demo2", str(img_idx)+".png")
    img_fname = os.path.join(FLAGS.logdir, "video_demo2_demo2", str(img_idx)+".jpg")

    color_image = cv2.cvtColor(cv2.imread(img_fname), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_fname,-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    cam_pose = np.array(frame['transform_matrix']) @ blender2opencv
    fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
    cam_intr = get_intrinsics(fx, fy, cx, cy)

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh_ddp_outdomain.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc_ddp_outdomain.ply", point_cloud)