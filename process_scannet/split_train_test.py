import argparse
import os, sys
import numpy as np

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--test_step', type=int, default=8, help='export every nth frame')

opt = parser.parse_args()

all_ids = sorted([int(d[:-4]) for d in os.listdir(os.path.join(opt.output_path, "rgb")) if ".jpg" in d])

test_every_step = opt.test_step


### Actual split
train_ids = []
test_ids = []
for i in range(len(all_ids)):
	if i%test_every_step == 0:
		test_ids.append(all_ids[i])
	else:
		train_ids.append(all_ids[i])

# ##Debugging
# train_ids = all_ids
# test_ids = all_ids


print(len(train_ids))
print(len(test_ids))

split_outpath = os.path.join(opt.output_path, "test_step_"+str(test_every_step))
if not os.path.exists(split_outpath):
    os.makedirs(split_outpath)

outfile = os.path.join(split_outpath, "train.txt")
with open(outfile, 'w') as f:
    for line in train_ids:
        f.write(f"{line}\n")

outfile = os.path.join(split_outpath, "test.txt")
with open(outfile, 'w') as f:
    for line in test_ids:
        f.write(f"{line}\n")


##### From scannet instantngp debugging
# ### Debugging
# outfile = os.path.join(opt.output_path, "train.txt")
# with open(outfile, 'w') as f:
#     for line in all_ids:
#         f.write(f"{line}\n")

# outfile = os.path.join(opt.output_path, "test.txt")
# with open(outfile, 'w') as f:
#     for line in all_ids:
#         f.write(f"{line}\n")