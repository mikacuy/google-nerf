import os, sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--video_1', default= "/orion/u/mikacuy/coordinate_mvs/piecewise_linear/log_blender_vanilla_importancesample_fullres/lego_piecewise_constant/0.mp4", type=str)
parser.add_argument('--video_2', default= "/orion/u/mikacuy/coordinate_mvs/piecewise_linear/log_blender_vanilla_importancesample_fullres/lego_piecewise_linear/0.mp4", type=str)

parser.add_argument('--out_name', default= "lego_importancesample_fullres_constantvslinear.mp4", type=str)


parser.add_argument('--video_3', default= "log_fewer_images_16v2/0830_scene0710_ddp/video_0/video.mp4", type=str)


parser.add_argument("--num_vids", type=int, default=2, help='number of video: current 2 or 3')

FLAGS = parser.parse_args()

## This should be working
if FLAGS.num_vids == 2:
	command = "ffmpeg \
		-i " + FLAGS.video_1 + \
		" -i " + FLAGS.video_2 + \
		" -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
		-map '[vid]' \
		-c:v libx264 \
		-crf 23 \
		-preset veryfast " +\
		FLAGS.out_name

	os.system(command)
	print("Combined 2 videos.")

### The three video version isn't stable last I checked
if FLAGS.num_vids == 3:
	command = "ffmpeg -i " + \
	FLAGS.video_1 + " -i " +\
	FLAGS.video_2 + " -i " + \
	FLAGS.video_3 + \
	" -filter_complex \"[1:v][0:v]scale2ref=oh*mdar:ih[1v][0v];[2:v][0v]scale2ref=oh*mdar:ih[2v][0v];[0v][1v][2v]hstack=3,scale='2*trunc(iw/2)':'2*trunc(ih/2)'\" "+\
	FLAGS.out_name

	print(command)

	os.system(command)
	print("Combined 3 videos.")	

print("Done.")