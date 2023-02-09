import cv2
import numpy as np
import os, sys
from model import to8b
import argparse
import matplotlib.pyplot as plt

###For parallel running
import datetime
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument('--all_logdirs', default='/orion/u/mikacuy/coordinate_mvs/dense_depth_priors_nerf/log_rebuttal_sparsity_corrected_marginal/', help='Root dir for dataset')
FLAGS = parser.parse_args()

ALL_LOGDIRS = FLAGS.all_logdirs

folders = os.listdir(ALL_LOGDIRS)
folders = sorted(folders)
print(len(folders))

commands = []
for logdir in folders:
	## Copy the original set of gt depth maps
	cmd = "python compute_LPIPS_single_log.py --logdir=" + os.path.join(ALL_LOGDIRS, logdir)
	commands.append(cmd)

######Run deformations in parallel#####
print("Number of commands in parallel: "+str(len(commands)))

report_step = 8
pool = Pool(report_step)
for idx, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
  if idx % report_step == 0:
     print('[%s] command %d of %d' % (datetime.datetime.now().time(), idx, len(commands)))
  if return_code != 0:
     print('!! command %d of %d (\"%s\") failed' % (idx, len(commands), commands[idx]))
################