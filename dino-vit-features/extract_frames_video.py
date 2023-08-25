import sys, os
import cv2

DUMP_DIR = "ground_tianhang_images"
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

vidcap = cv2.VideoCapture('ground_tianhang.mp4')
success,image = vidcap.read()
count = 0
while success:
  if count % 100 == 0:
    cv2.imwrite(os.path.join(DUMP_DIR, "frame%d.jpg" % count), image)     # save frame as JPEG file     

  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1