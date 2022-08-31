import os, sys
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scenename', default='Lego', help='Dataset loader name')
parser.add_argument('--dataroot', default='/orion/group/NSVF/Synthetic_NeRF/', help='Root dir for dataset')
parser.add_argument('--depthpath', default='/orion/u/mikacuy/coordinate_mvs/ngp_pl/results/nsvf/Lego/', help='Root dir for dataset')

FLAGS = parser.parse_args()

SCENENAME = FLAGS.scenename
DATAROOT = FLAGS.dataroot
ROOT_DIR = os.path.join(DATAROOT, SCENENAME)
DEPTHPATH = FLAGS.depthpath

outfol_fol = os.path.join(ROOT_DIR, "leres_cimle_v1")
if not os.path.exists(outfol_fol): os.mkdir(outfol_fol)

### Copy rgb images
prefix = '2_'
imgs = sorted(glob.glob(os.path.join(ROOT_DIR, 'rgb', prefix+'*.png')))
depth_maps = sorted(glob.glob(os.path.join(DEPTHPATH, '*_d.png')))

if len(imgs) !=  len(depth_maps):
	print("ERROR. Number of images and depth maps don't match.")
	exit()

print(len(imgs))
print(len(depth_maps))

### Create rgb and depth folder
rgb_fol = os.path.join(outfol_fol, "rgb")
if not os.path.exists(rgb_fol): os.mkdir(rgb_fol)

depth_fol = os.path.join(outfol_fol, "depth")
if not os.path.exists(depth_fol): os.mkdir(depth_fol)

for i in range(len(imgs)):
	### Copy files in order
	rgb_fname = imgs[i]
	out_rgb_fname = os.path.join(rgb_fol, str(i).zfill(4)+"_rgb.png")
	os.system('cp %s %s' % (rgb_fname, out_rgb_fname))

	depth_fname = depth_maps[i]
	out_depth_fname = os.path.join(depth_fol, str(i).zfill(4)+"_depth.png")
	os.system('cp %s %s' % (depth_fname, out_depth_fname))