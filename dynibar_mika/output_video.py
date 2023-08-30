import subprocess
import os, sys
import numpy as np
import math
import time
import cv2


ref_frame = 6
cam_idx = 11

curr_fname = str(ref_frame) + "_" + str(cam_idx)

SOURCE_DIR = "/home/mikacuy/Desktop/coord-mvs/google-nerf/dynibar_mika/Nvidia/balloon1/"
DUMP_DIR = os.path.join(SOURCE_DIR, "video_flipped")
os.makedirs(DUMP_DIR, exist_ok=True)

curr_outdir = os.path.join(DUMP_DIR, curr_fname)
os.makedirs(curr_outdir, exist_ok=True)

gt_fnames = []
pred_fnames = []
flow_fnames = []

neighbor_idx = [-3, -2, -1, 0, 1, 2, 3]

for idx in neighbor_idx:
    n_frame_idx = ref_frame + idx

    ### At that particular time, get the same camera
    n_fname = str(n_frame_idx) + "_" + str(cam_idx)

    n_gt_name = os.path.join(SOURCE_DIR, "renderings", n_fname + "_1gt.jpg")
    n_pred_fname = os.path.join(SOURCE_DIR, "renderings", n_fname + "_0rgb.jpg")
    # n_flow_fname = os.path.join(SOURCE_DIR, "pred_flow", curr_fname + "_flow" + str(idx)+".jpg")
    n_flow_fname = os.path.join(SOURCE_DIR, "pred_flow_flipped", curr_fname + "_flow" + str(idx)+".jpg")

    gt_fnames.append(n_gt_name)
    pred_fnames.append(n_pred_fname)
    flow_fnames.append(n_flow_fname)


### Write to file for ffmpeg
gt_file = open(os.path.join(curr_outdir, "gt_fnames.txt"),'w')
for item in gt_fnames:
    # gt_file.write("file \'" + item + "\'\n")
    gt_file.write("file " + item + "\n")
    gt_file.write("duration 0.5 \n")
gt_file.close()

pred_file = open(os.path.join(curr_outdir, "pred_fnames.txt"),'w')
for item in pred_fnames:
    # pred_file.write("file \'" + item + "\'\n")
    pred_file.write("file " + item + "\n")
    pred_file.write("duration 0.5 \n")
pred_file.close()

flow_file = open(os.path.join(curr_outdir, "flow_fnames.txt"),'w')
for item in flow_fnames:
    # flow_file.write("file \'" + item + "\'\n")
    flow_file.write("file " + item + "\n")
    flow_file.write("duration 0.5 \n")
flow_file.close()

### ffmpeg to video
fps = 2
gt_video_file = os.path.join(curr_outdir, "gt_rgb.mp4")
subprocess.call(["ffmpeg", "-y",  "-f", "concat", "-safe", "0", "-i", os.path.join(curr_outdir, "gt_fnames.txt"), "-c:v", "libx264", "-profile:v", "high", "-crf", str(fps), gt_video_file])

rgb_video_file = os.path.join(curr_outdir, "pred_rgb.mp4")
subprocess.call(["ffmpeg", "-y",  "-f", "concat", "-safe", "0", "-i", os.path.join(curr_outdir, "pred_fnames.txt"), "-c:v", "libx264", "-profile:v", "high", "-crf", str(fps), rgb_video_file])

flow_video_file = os.path.join(curr_outdir, "pred_trj_flow.mp4")
subprocess.call(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(curr_outdir, "flow_fnames.txt"), "-c:v", "libx264", "-profile:v", "high", "-crf", str(fps), flow_video_file])


out_fname = os.path.join(curr_outdir, "combined.mp4")

command = "ffmpeg -i " + \
gt_video_file + " -i " +\
rgb_video_file + " -i " + \
flow_video_file + \
" -filter_complex \"[1:v][0:v]scale2ref=oh*mdar:ih[1v][0v];[2:v][0v]scale2ref=oh*mdar:ih[2v][0v];[0v][1v][2v]hstack=3,scale='2*trunc(iw/2)':'2*trunc(ih/2)'\" "+\
out_fname

print(command)

os.system(command)
print("Combined 3 videos.")	

# command = "ffmpeg \
#     -i " + gt_video_file + \
#     " -i " + flow_video_file + \
#     " -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
#     -map '[vid]' \
#     -c:v libx264 \
#     -crf 23 \
#     -preset veryfast " +\
#     FLAGS.out_name

# os.system(command)
# print("Combined 2 videos.")


