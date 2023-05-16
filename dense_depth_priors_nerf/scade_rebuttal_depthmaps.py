import cv2
import numpy as np
import os, sys
# from model import to8b
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/scenes/scene0758_00/', help='Root dir for dataset')
parser.add_argument('--dump_dir', default= "scade_rebuttal_depthmaps_scene758/", type=str)
FLAGS = parser.parse_args()

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

DATA_DIR = FLAGS.dataroot

depthfile1 = "/orion/u/mikacuy/coordinate_mvs/dense_depth_priors_nerf/log_scene758_videodemo_finetune/scene0758_freezess_lr5e4_clr5e6/test_images_scene0758_00/0_d.png"
depthfile2 = "/orion/u/mikacuy/coordinate_mvs/dense_depth_priors_nerf/Scannet/scene758/scene0758_00_ddp_outdomain/test_images_scene0758_00/0_d.png"
depthfile3 = os.path.join(DATA_DIR, "test/target_depth/232.png")
img_file = os.path.join(DATA_DIR, "test/rgb/232.jpg")

# out_img_fname = os.path.join(DUMP_DIR, "rgb.jpg")
# cmd = "cp "+ img_file + " " + out_img_fname
# os.system(cmd)

# depth1 = cv2.imread(depthfile1, cv2.IMREAD_UNCHANGED)
# print(depth1)
# plt.imsave(os.path.join(DUMP_DIR, "scade.jpg"), depth1, cmap='rainbow')
# # depth_colored_frame1 = cv2.applyColorMap(depth1, cv2.COLORMAP_TURBO)
# # cv2.imwrite(os.path.join(DUMP_DIR, "scade.jpg"), depth_colored_frame1)

# depth2 = cv2.imread(depthfile2, cv2.IMREAD_UNCHANGED)
# print(depth2)
# plt.imsave(os.path.join(DUMP_DIR, "ddp_outdomain.jpg"), depth2, cmap='rainbow')
# # depth_colored_frame2 = cv2.applyColorMap(depth2, cv2.COLORMAP_TURBO)
# # cv2.imwrite(os.path.join(DUMP_DIR, "ddp_outdomain.jpg"), depth_colored_frame2)

# depth3 = cv2.imread(depthfile3, cv2.IMREAD_UNCHANGED)

# ### Interpolate the zero values
# from scipy.interpolate import griddata
# H, W = depth3.shape

# grid_x,grid_y = np.meshgrid(np.linspace(0, H-1, H), np.linspace(0, W-1, W), indexing='ij')

# coords = np.stack(np.meshgrid(np.linspace(0, H-1, H), np.linspace(0, W-1, W), indexing='ij'), -1) 
# coords = np.reshape(coords, [-1,2])  # (H * W, 2)
# idx_valid = depth3!=0
# idx_valid = np.reshape(idx_valid, [-1])  # (H * W, 2)
# coords = coords[idx_valid]

# depth3_flatten = np.reshape(depth3, [-1])  # (H * W, 2)
# depth3_flatten_valid = depth3_flatten[idx_valid]

# print(coords.shape)
# print(depth3_flatten_valid.shape)

# depth3 = griddata(coords, depth3_flatten_valid, (grid_x, grid_y), method='nearest', rescale=False)

# print(depth3)
# plt.imsave(os.path.join(DUMP_DIR, "gt.jpg"), depth3, cmap='rainbow')
# # depth_colored_frame3 = cv2.applyColorMap(depth3, cv2.COLORMAP_TURBO)
# # cv2.imwrite(os.path.join(DUMP_DIR, "gt.jpg"), depth_colored_frame3)


##### Output fusion error
depth3 = cv2.imread(depthfile3, cv2.IMREAD_UNCHANGED)

### Interpolate the zero values
from scipy.interpolate import griddata
H, W = depth3.shape

grid_x,grid_y = np.meshgrid(np.linspace(0, H-1, H), np.linspace(0, W-1, W), indexing='ij')

coords = np.stack(np.meshgrid(np.linspace(0, H-1, H), np.linspace(0, W-1, W), indexing='ij'), -1) 
coords = np.reshape(coords, [-1,2])  # (H * W, 2)
idx_valid = depth3!=0
idx_valid = np.reshape(idx_valid, [-1])  # (H * W, 2)
coords = coords[idx_valid]

depth3_flatten = np.reshape(depth3, [-1])  # (H * W, 2)
depth3_flatten_valid = depth3_flatten[idx_valid]

depth3 = griddata(coords, depth3_flatten_valid, (grid_x, grid_y), method='nearest', rescale=False)


fusion_depth_file = "outputs_for_mika_v2.npz"
print(fusion_depth_file)

with open('outputs_for_mika_v2.npz', 'rb') as f:
	data = np.load(f)

	color_ddp_fusion = data.files[0]
	color_scade_fusion = data.files[1]
	depth_ddp_fusion = data.files[2]
	depth_scade_fusion = data.files[3]

	print(color_ddp_fusion)


print(depth_scade_fusion.max())
print(depth_ddp_fusion.max())
print()
print(depth_scade_fusion.min())
print(depth_ddp_fusion.min())




















