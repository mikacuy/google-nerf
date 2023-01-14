import os, sys
import math
import numpy as np 
import time
import datetime

import cv2
from data import load_scene_mika, load_scene_processed
import trimesh
import matplotlib


cmap = matplotlib.cm.get_cmap('plasma')

def point_cloud_from_depth(depth, intrinsic):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """

    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]

    H, W = depth.shape
    c, r = np.meshgrid(np.arange(W), np.arange(H), sparse=True)
    valid = (depth > 0) & (depth < 8)
    z = -depth
    x = depth * ((c + 0.5)-cx)/fx
    y = depth * (H - (r + 0.5) - cy)/fy
    x = x[valid]
    y = y[valid]
    z = z[valid]
    return np.dstack((x, y, z))

# DUMP_DIR = "plot_prior_depth_scene710"
# scene_data_dir = os.path.join("/orion/group/scannet_v2/dense_depth_priors/scenes/", "scene0710_00")
# cimle_dir = "dump_1009_pretrained_dd_scene0710_train_unifiedscale_rotated/"

# DUMP_DIR = "plot_prior_depth_scene758"
# scene_data_dir = os.path.join("/orion/group/scannet_v2/dense_depth_priors/scenes/", "scene0758_00")
# cimle_dir = "dump_1009_pretrained_dd_scene0758_train_unifiedscale_rotated/"

# DUMP_DIR = "plot_prior_depth_scene710_bigsubset"
# scene_data_dir = os.path.join("/orion/group/scannet_v2/dense_depth_priors/scenes/", "scene0710_00")
# cimle_dir = "dump_1009_pretrained_dd_scene0710_train_unifiedscale_rotated_bigsubset_dataparallel/"

DUMP_DIR = "plot_prior_depth_Auditorium"
scene_data_dir = os.path.join("/orion/u/mikacuy/coordinate_mvs/processed_scenes/", "Auditorium_subsample")
cimle_dir = "dump_1107_Auditoriumsubsample_sfmaligned_indv/"

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

num_hypothesis = 20

images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
gt_depths, gt_valid_depths, all_depth_hypothesis = load_scene_processed(scene_data_dir, cimle_dir, num_hypothesis, 'transforms_train.json')

# images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
# gt_depths, gt_valid_depths, all_depth_hypothesis = load_scene_mika(scene_data_dir, cimle_dir, num_hypothesis, 'transforms_train.json')

i_train, i_val, i_test, i_video = i_split

images = images[i_train]
depths = depths[i_train]

# gt_depths = gt_depths[i_train]
# gt_depths = gt_depths[i_train]

poses = poses[i_train]
intrinsics = intrinsics[i_train]

print(images.shape)
# print(gt_depths.shape)

print(depths.shape)
print(all_depth_hypothesis.shape)
print(poses.shape)
print(intrinsics.shape)


### Reconstruct ground truth depth
for i in range(len(poses)):
	c2w = poses[i]
	# depth_img = gt_depths[i].squeeze()
	depth_img = depths[i].squeeze()

	pc = point_cloud_from_depth(depth_img, intrinsics[i]).reshape(-1, 3)

	### Transform point cloud
	R = c2w[:3,:3]
	t = c2w[:3,3]

	pc =  pc @ R.T + t 

	rgba = cmap(np.ones(pc.shape[0])*float(i)/len(poses))
	cloud=trimesh.PointCloud(pc, colors=rgba)

	# cloud.pointColors(np.ones(pc.shape[0])*float(i), cmap='viridis')

	cloud.export(os.path.join(DUMP_DIR, 'gt_'+str(i)+'.ply'))

print("Done with ground truth.")


### Reconstruct hypotheses
for i in range(len(poses)):
	c2w = poses[i]

	for j in range(num_hypothesis):

		depth_img = all_depth_hypothesis[i][j].squeeze()

		pc = point_cloud_from_depth(depth_img, intrinsics[i]).reshape(-1, 3)

		### Transform point cloud
		R = c2w[:3,:3]
		t = c2w[:3,3]

		pc =  pc @ R.T + t 

		rgba = cmap(np.ones(pc.shape[0])*float(j)/num_hypothesis)
		cloud=trimesh.PointCloud(pc, colors=rgba)

		# cloud.pointColors(np.ones(pc.shape[0])*float(i), cmap='viridis')

		cloud.export(os.path.join(DUMP_DIR, 'leres_'+str(i)+'_'+str(j)+'.ply'))


print("Done with LeReS outputs.")









