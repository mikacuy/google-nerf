import os, sys
import imageio
import glob
import argparse
import json

BASE_DIR = "/orion/group/scannet_v2/dense_depth_priors/scenes/"

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default= "scene0710_00/", type=str)
parser.add_argument('--outfile', default= "transforms_train_16_v2.json", type=str)

FLAGS = parser.parse_args()
INPUT_DIR = FLAGS.input_dir
OUTFILE = FLAGS.outfile

with open(os.path.join(BASE_DIR, INPUT_DIR, 'transforms_train.json'), 'r') as fp:
    meta = json.load(fp)

near = float(meta['near'])
far = float(meta['far'])
depth_scaling_factor = float(meta['depth_scaling_factor'])

### Removed for scene0710_00 transforms_train_16_v1.json
# files_to_remove = ["1185.jpg", "1712.jpg"]

### Removed for scene0710_00 transforms_train_16_v2.json
files_to_remove = ["1759.jpg", "1000.jpg"]

pruned = []

for frame in meta['frames']:
	if frame["file_path"].split("/")[-1] in files_to_remove:
		continue
	pruned.append(frame)

meta['frames'] = pruned

out_file = open(os.path.join(BASE_DIR, INPUT_DIR, OUTFILE), "w")
json.dump(meta, out_file, indent = 6)  
out_file.close()

with open(os.path.join(BASE_DIR, INPUT_DIR, OUTFILE), 'r') as fp:
    meta = json.load(fp)
print(len(meta["frames"]))