import os, sys
import imageio
import glob
import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--image_dir', default= "log_scene0738/0902_scene0738_norm_001/video_0/", type=str)
# parser.add_argument('--image_dir', default= "log_scene0738_00_vanillanerf/0902_scene0738/video_0/", type=str)
# parser.add_argument('--image_dir', default= "log_scene0738_00_ddp/0902_scene0738/video_0/", type=str)

# parser.add_argument('--image_dir', default= "log_scene0758/0830_scene0758_norm_004/video_0/", type=str)
# parser.add_argument('--image_dir', default= "log_scene0758_00_vanilla_nerf/20220826_152320_scene0758_00/video_0/", type=str)
# parser.add_argument('--image_dir', default= "log_scene0758_00/20220826_152210_scene0758_00/video_0/", type=str)


# parser.add_argument('--image_dir', default= "log_scene0781/0830_scene0781_norm_003/video_0/", type=str)
# parser.add_argument('--image_dir', default= "log_scene0781_00_vanilla_nerf/20220826_152524_scene0781_00/video_0/", type=str)
parser.add_argument('--image_dir', default= "log_scene0781_00/20220826_152225_scene0781_00/video_0/", type=str)

FLAGS = parser.parse_args()
IMAGE_DIR = FLAGS.image_dir


img_paths = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))

frames = []

for img_path in img_paths:
	frame = int(img_path.split("/")[-1][:-4])
	frames.append(frame)

frames = sorted(frames)

imgs = []

for frame in frames:
	img_path = os.path.join(IMAGE_DIR, str(frame)+".jpg")
	imgs.append(img_path)

imageio.mimsave(os.path.join(IMAGE_DIR, 'video.mp4'),
                [imageio.imread(img) for img in imgs],
                fps=10, macro_block_size=1)

print("Done processing video.")