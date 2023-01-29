import cv2
import numpy as np
import os, sys
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', default= "scade_rebuttal_nonopaque_surfaces/", type=str)
# parser.add_argument('--dump_dir', default= "scade_rebuttal_reflective_surfaces_v2/", type=str)
FLAGS = parser.parse_args()

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

## Non-opaque surfaces
filenames_depth = ["/orion/downloads/coordinate_mvs/taskonomy/dump_taskonomy_visu_v2/adairsville_point_1074_view_3_domain_depth_zbuffer.png", \
			"/orion/downloads/coordinate_mvs/taskonomy/dump_taskonomy_visu_v2/alfred_point_1002_view_2_domain_depth_zbuffer.png", \
			"/orion/downloads/coordinate_mvs/taskonomy/dump_taskonomy_visu_v2/anaheim_point_100_view_6_domain_depth_zbuffer.png", \
			"/orion/downloads/coordinate_mvs/taskonomy/dump_taskonomy_visu_v2/anaheim_point_1023_view_1_domain_depth_zbuffer.png", \
			"/orion/downloads/coordinate_mvs/taskonomy/dump_taskonomy_visu_v2/adairsville_point_1073_view_3_domain_depth_zbuffer.png"]

filenames_rgb = ["/orion/downloads/coordinate_mvs/taskonomy/rgbs/adairsville/point_1074_view_3_domain_rgb.png", \
			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/alfred/point_1002_view_2_domain_rgb.png", \
			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/anaheim/point_100_view_6_domain_rgb.png", \
			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/anaheim/point_1023_view_1_domain_rgb.png", \
			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/adairsville/point_1073_view_3_domain_rgb.png"]

# ### Reflective surfaces
# filenames_rgb = ["/orion/downloads/coordinate_mvs/taskonomy/rgbs/almena/point_1006_view_0_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/aldrich/point_1008_view_14_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/almena/point_1017_view_2_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/almena/point_1027_view_2_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/american/point_1049_view_3_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/alstown/point_1096_view_4_domain_rgb.png"]

# ### Reflective surfaces
# filenames_rgb = ["/orion/downloads/coordinate_mvs/taskonomy/rgbs/badger/point_611_view_6_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/belpre/point_1401_view_4_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/belpre/point_305_view_9_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/belpre/point_464_view_1_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/belpre/point_1049_view_3_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/benicia/point_257_view_0_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/bertram/point_104_view_5_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/bertram/point_538_view_8_domain_rgb.png", \
# 			"/orion/downloads/coordinate_mvs/taskonomy/rgbs/alstown/point_900_view_10_domain_rgb.png"]

for i in range(len(filenames_rgb)):
	fname_rgb = filenames_rgb[i]
	
	scene_name = fname_rgb.split("/")[-2]
	out_img_fname = os.path.join(DUMP_DIR, scene_name+ "_" +fname_rgb.split("/")[-1])
	print(out_img_fname)
	cmd = "cp "+ fname_rgb + " " + out_img_fname
	os.system(cmd)


	fname_depth = filenames_depth[i]

	out_img_fname = os.path.join(DUMP_DIR, scene_name+ "_" +fname_depth.split("/")[-1])
	print(out_img_fname)
	cmd = "cp "+ fname_depth + " " + out_img_fname
	os.system(cmd)
