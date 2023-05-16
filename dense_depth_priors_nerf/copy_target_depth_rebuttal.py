import cv2
import numpy as np
import os, sys
from model import to8b
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/', help='Root dir for dataset')
parser.add_argument('--ddp_root', default='/orion/group/scannet_v2/dense_depth_priors/scenes/scene0781_00/', help='Root dir for dataset')
parser.add_argument('--scannet_traj_path', default='/orion/group/scannet_v2/scade_rebuttal/scene0781_00/', help='Root dir for dataset')
FLAGS = parser.parse_args()

DATA_DIR = FLAGS.dataroot
DDP_DIR = FLAGS.ddp_root
SCANNET_PATH = FLAGS.scannet_traj_path
ADDTL_VIEWS_DEPTH_DIR = os.path.join(SCANNET_PATH, "depth")

folders = os.listdir(DATA_DIR)
folders = sorted(folders)
print(folders)


for fol in folders:
	print("Processing "+fol+".")
	curr_dir = os.path.join(DATA_DIR, fol)
	target_depth_fol = os.path.join(curr_dir, "train", "target_depth")

	## Copy the original set of gt depth maps
	cmd = "cp -r " + os.path.join(DDP_DIR, "train", "target_depth") + " " + os.path.join(curr_dir, "train")
	os.system(cmd)

	rgb_files = os.listdir(os.path.join(curr_dir, "train", "rgb"))
	fnames = [x[:-4] for x in rgb_files]
	depth_files = [x+".png" for x in fnames]
	
	for i in range(len(depth_files)):
		depth_file = depth_files[i]
		if depth_file not in os.listdir(target_depth_fol):
			cmd = "cp " + os.path.join(ADDTL_VIEWS_DEPTH_DIR, depth_file) + " " + os.path.join(target_depth_fol, depth_file)
			os.system(cmd)
			print("Copied "+depth_file+".")

	print()