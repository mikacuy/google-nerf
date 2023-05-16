import sys, os
import datetime

from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

DUMP_DIR = "/home/mikacuy/Desktop/coord-mvs/LeReS_data/taskonomy/tmp/rgbs"
os.makedirs(DUMP_DIR, exist_ok=True)

DATA_DIR = "/home/mikacuy/Desktop/coord-mvs/LeReS_data/taskonomy/rgbs"

scenes = os.listdir(DATA_DIR)

print(scenes)
print(len(scenes))


commands = []
for scene in scenes:
	src_path = os.path.join(DATA_DIR, scene)

	c = "cp -r " + src_path + " " + DUMP_DIR
	commands.append(c)

print("Copying scenes for colmap tmp")
print("Number of commands in parallel: "+str(len(commands)))

report_step = 24
pool = Pool(report_step)
for idx, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
  if idx % report_step == 0:
     print('[%s] command %d of %d' % (datetime.datetime.now().time(), idx, len(commands)))
  if return_code != 0:
     print('!! command %d of %d (\"%s\") failed' % (idx, len(commands), commands[idx]))