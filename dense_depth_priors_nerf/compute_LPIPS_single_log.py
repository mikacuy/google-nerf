import cv2
import numpy as np
import os, sys
import argparse
from lpips import LPIPS

from data import load_scene
import json
from argparse import Namespace
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='/orion/u/mikacuy/coordinate_mvs/dense_depth_priors_nerf/log_rebuttal_sparsity_corrected/scene0781_1_scade', help='Root dir for dataset')

parser.add_argument('--data_root', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781', help='Root dir for dataset')
FLAGS = parser.parse_args()

DATA_ROOT = FLAGS.data_root
args_file = os.path.join(FLAGS.logdir, 'args.json')
print(FLAGS.logdir)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

with open(args_file, 'r') as af:
    args_dict = json.load(af)
args = Namespace(**args_dict)

scene_data_dir = os.path.join(DATA_ROOT, args.scene_id)

images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
gt_depths, gt_valid_depths = load_scene(scene_data_dir, 'transforms_train.json')

i_train, i_val, i_test, i_video = i_split

lpips_alex = LPIPS()
lpips_alex = lpips_alex.to(device)

images = torch.Tensor(images[i_test]).to(device)
result_dir = os.path.join(FLAGS.logdir, "test_images_" + args.scene_id)

lpips_all = []
for n in range(images.shape[0]):
	curr_output_rgb_file = os.path.join(result_dir, str(n) + "_rgb" + ".jpg")
	rgb = cv2.imread(curr_output_rgb_file, cv2.IMREAD_UNCHANGED)
	convert_fn = cv2.COLOR_BGR2RGB
	rgb = (cv2.cvtColor(rgb, convert_fn) / 255.).astype(np.float32)

	rgb = torch.Tensor(rgb).to(device)
	target = images[n].to(device)

	lpips = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]

	lpips_all.append(lpips[0, 0, 0].detach().cpu().numpy())

lpips_all = np.array(lpips_all)
mean_lpips = np.mean(lpips_all)

print(lpips_all.shape)

print(mean_lpips)

LOG_FOUT = open(os.path.join(result_dir, 'lpips.txt'), 'w')
LOG_FOUT.write("Average LPIPS:"+'\n')
LOG_FOUT.write(str(mean_lpips)+'\n')
LOG_FOUT.close()



