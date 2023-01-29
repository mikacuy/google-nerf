import os
import random
import json

import numpy as np
import torch
from torchvision import transforms

import skimage
import skimage.io
import matplotlib.pyplot as plt
import cv2


dataset_dir = "/orion/downloads/coordinate_mvs/"
data_split = "test"

dir_anno = os.path.join(dataset_dir, "taskonomy",
                      'annotations',
                      data_split + '_annotations.json')

with open(dir_anno, 'r') as load_f:
    all_annos = json.load(load_f)

rgb_files = [
    os.path.join(dataset_dir, all_annos[i]['rgb_path']) 
    for i in range(len(all_annos))
]
depth_files = [
    os.path.join(dataset_dir, all_annos[i]['depth_path'])
    if 'depth_path' in all_annos[i]
    else None
    for i in range(len(all_annos))
]

DUMP_DIR = "/orion/downloads/coordinate_mvs/taskonomy/dump_taskonomy_visu_test"
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

for i in range(len(depth_files)):
	fname_depth = depth_files[i]
	fname_rgb = rgb_files[i]

	depth = cv2.imread(fname_depth, cv2.IMREAD_UNCHANGED)
	depth[depth > 23000] = 0
	drange = 512.0
	depth = depth / drange

	scene_name = fname_depth.split("/")[-2]
	out_depth_fname = os.path.join(DUMP_DIR, scene_name+ "_" + fname_depth.split("/")[-1])

	plt.clf()
	plt.figure()
	plt.imsave(out_depth_fname, depth, cmap='rainbow') 
	# ax = plt.imshow(depth, cmap="rainbow")
	# plt.colorbar(ax)
	# plt.savefig(out_depth_fname)

	out_img_fname = os.path.join(DUMP_DIR, scene_name + "_" + fname_rgb.split("/")[-1])
	cmd = "cp "+ fname_rgb + " " + out_img_fname
	os.system(cmd)

	if i%1000 == 0:
		print("Finished "+str(i)+"/"+str(len(depth_files))+".")




# filenames_depth = ["/orion/downloads/coordinate_mvs/taskonomy/depths/burien/point_1363_view_2_domain_depth_zbuffer.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/depths/haymarket/point_1396_view_1_domain_depth_zbuffer.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/depths/clarkridge/point_758_view_3_domain_depth_zbuffer.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/depths/markleeville/point_693_view_2_domain_depth_zbuffer.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/depths/arkansaw/point_1184_view_6_domain_depth_zbuffer.png"]

# filenames_rgb = ["/orion/downloads/coordinate_mvs/taskonomy/rgbs/burien/point_1363_view_2_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/haymarket/point_1396_view_1_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/clarkridge/point_758_view_3_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/markleeville/point_693_view_2_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/arkansaw/point_1184_view_6_domain_rgb.png"]

# DUMP_DIR = "dump_debug_taskonomy"
# if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)



# for i in range(len(filenames_depth)):
# 	fname_depth = filenames_depth[i]
# 	fname_rgb = filenames_rgb[i]

# 	# depth = skimage.io.imread(fname_depth)
# 	depth = cv2.imread(fname_depth, cv2.IMREAD_UNCHANGED)

# 	depth[depth > 23000] = 0
# 	drange = 512.0
# 	depth = depth / drange


# 	out_depth_fname = os.path.join(DUMP_DIR, fname_depth.split("/")[-1])
# 	print(out_depth_fname)

# 	plt.clf()
# 	plt.figure()
# 	ax = plt.imshow(depth, cmap="rainbow")
# 	plt.colorbar(ax)
# 	plt.savefig(out_depth_fname)


# 	out_img_fname = os.path.join(DUMP_DIR, fname_rgb.split("/")[-1])
# 	print(out_img_fname)
# 	cmd = "cp "+ fname_rgb + " " + out_img_fname
# 	os.system(cmd)
