import os, sys
import math
import numpy as np 
import time
import datetime

import cv2
from data import load_scene_mika
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

def point_cloud_from_depth_taskonomy(depth, intrinsic):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """

    fx, fy, cx, cy = 512., 512., depth.shape[1]/2., depth.shape[0]/2.
    # print(intrinsic)
    # print(fx)
    # print(fy)
    # print(cx)
    # print(cy)
    # exit()

    H, W = depth.shape
    c, r = np.meshgrid(np.arange(W), np.arange(H), sparse=True)
    valid = (depth > 0) & (depth < 8)
    z = -depth
    # x = depth * ((c + 0.5)-cx)/fx
    # y = depth * (H - (r + 0.5) - cy)/fy
    x = depth * ((c )-cx)/fx
    y = depth * (H - (r ) - cy)/fy    
    x = x[valid]
    y = y[valid]
    z = z[valid]
    return np.dstack((x, y, z))

# def reconstruct_3D(depth, f):
#     """
#     Reconstruct depth to 3D pointcloud with the provided focal length.
#     Return:
#         pcd: N X 3 array, point cloud
#     """
#     cu = depth.shape[1] / 2
#     cv = depth.shape[0] / 2
#     width = depth.shape[1]
#     height = depth.shape[0]
#     row = np.arange(0, width, 1)
#     u = np.array([row for i in np.arange(height)])
#     col = np.arange(0, height, 1)
#     v = np.array([col for i in np.arange(width)])
#     v = v.transpose(1, 0)

#     if f > 1e5:
#         print('Infinite focal length!!!')
#         x = u - cu
#         y = v - cv
#         z = depth / depth.max() * x.max()
#     else:
#         x = (u - cu) * depth / f
#         y = (v - cv) * depth / f
#         z = depth

#     x = np.reshape(x, (width * height, 1)).astype(np.float)
#     y = np.reshape(y, (width * height, 1)).astype(np.float)
#     z = np.reshape(z, (width * height, 1)).astype(np.float)
#     pcd = np.concatenate((x, y, z), axis=1)
#     # pcd = pcd.astype(np.int)
#     return pcd



# DUMP_DIR = "plot_prior_depth_scene710"
# scene_data_dir = os.path.join("/orion/group/scannet_v2/dense_depth_priors/scenes/", "scene0710_00")
# cimle_dir = "dump_1009_pretrained_dd_scene0710_train_unifiedscale_rotated/"

# DUMP_DIR = "plot_prior_depth_scene758"
# scene_data_dir = os.path.join("/orion/group/scannet_v2/dense_depth_priors/scenes/", "scene0758_00")
# cimle_dir = "dump_1009_pretrained_dd_scene0758_train_unifiedscale_rotated/"

# DUMP_DIR = "plot_prior_depth_scene710_bigsubset"
# scene_data_dir = os.path.join("/orion/group/scannet_v2/dense_depth_priors/scenes/", "scene0710_00")
# cimle_dir = "dump_1009_pretrained_dd_scene0710_train_unifiedscale_rotated_bigsubset_dataparallel/"

DUMP_DIR = "plot_depth_hypothesis_sfm_scene758_gtss_taskonomy_intrinsics2"


scene_data_dir = os.path.join("/orion/group/scannet_v2/dense_depth_priors/scenes/", "scene0758_00")
cimle_dir = "dump_1009_pretrained_dd_scene0758_train_unscaled_rotated_bigsubset_dataparallel/"
scales_dir = "dump_1022_scene0758_scaleshift_0926big_dp_e56"

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

num_hypothesis = 20


# images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
# gt_depths, gt_valid_depths, all_depth_hypothesis, \
# scales_init, shifts_init = load_scene_mika(scene_data_dir, cimle_dir, num_hypothesis, 'transforms_train.json', True, scales_dir, False)

images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
gt_depths, gt_valid_depths, all_depth_hypothesis, \
scales_init, shifts_init = load_scene_mika(scene_data_dir, cimle_dir, num_hypothesis, 'transforms_train.json', True, scales_dir, True)


# print(intrinsics[0])
# print(images[0].shape)
# exit()

i_train, i_val, i_test, i_video = i_split

images = images[i_train]
gt_depths = gt_depths[i_train]
depths = depths[i_train]

poses = poses[i_train]
intrinsics = intrinsics[i_train]

print(images.shape)
print(gt_depths.shape)
print(all_depth_hypothesis.shape)
print(poses.shape)
print(intrinsics.shape)
print(scales_init)
print(shifts_init)
print(scales_init.shape)
print(shifts_init.shape)


### Reconstruct ground truth depth
for i in range(len(poses)):
	c2w = poses[i]
	R = c2w[:3,:3]
	t = c2w[:3,3]


	depth_img = gt_depths[i].squeeze()
	pc = point_cloud_from_depth(depth_img, intrinsics[i]).reshape(-1, 3)

	sfm_depth_img = depths[i].squeeze()
	pc_sfm = point_cloud_from_depth(sfm_depth_img, intrinsics[i]).reshape(-1, 3)

	### Transform point cloud
	pc =  pc @ R.T + t 

	rgba = cmap(np.ones(pc.shape[0])*float(i)/len(poses))
	cloud=trimesh.PointCloud(pc, colors=rgba)

	# cloud.pointColors(np.ones(pc.shape[0])*float(i), cmap='viridis')
	cloud.export(os.path.join(DUMP_DIR, 'gt_'+str(i)+'.ply'))


	pc_sfm =  pc_sfm @ R.T + t 

	rgba = cmap(np.ones(pc_sfm.shape[0])*float(i)/len(poses))
	cloud=trimesh.PointCloud(pc_sfm, colors=rgba)

	# cloud.pointColors(np.ones(pc.shape[0])*float(i), cmap='viridis')
	cloud.export(os.path.join(DUMP_DIR, 'gtsfm_'+str(i)+'.ply'))


print("Done with ground truth.")


### Reconstruct hypotheses
for i in range(len(poses)):
	c2w = poses[i]

	for j in range(num_hypothesis):

		curr_scale = scales_init[i]
		curr_shift = shifts_init[i]

		depth_img = all_depth_hypothesis[i][j].squeeze()
		depth_img = curr_scale*depth_img+curr_shift

		pc = point_cloud_from_depth_taskonomy(depth_img, intrinsics[i]).reshape(-1, 3)

		### Transform point cloud
		R = c2w[:3,:3]
		t = c2w[:3,3]

		pc =  pc @ R.T + t 

		rgba = cmap(np.ones(pc.shape[0])*float(j)/num_hypothesis)
		cloud=trimesh.PointCloud(pc, colors=rgba)

		# cloud.pointColors(np.ones(pc.shape[0])*float(i), cmap='viridis')

		cloud.export(os.path.join(DUMP_DIR, 'leres_'+str(i)+'_'+str(j)+'.ply'))


print("Done with LeReS outputs.")









