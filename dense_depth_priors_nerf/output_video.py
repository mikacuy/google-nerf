import os, sys
import imageio
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default= "log_space_carving_001/0829_scene0710/video_0/", type=str)

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