import os, sys
import argparse
import json
import numpy as np

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='taskonomy', help='dataset to use')
parser.add_argument('--data_split', type=str, default='train', help='data split')
parser.add_argument('--num_subset', type=int, default=116864, help='number of examples to sample')
parser.add_argument('--dataroot', default='/home/mikacuy/Desktop/coord-mvs/LeReS_data/', help='Root dir for dataset')

FLAGS = parser.parse_args()
DATASET_NAME = FLAGS.dataset_name
DATA_SPLIT = FLAGS.data_split
NUM_SUBSET = FLAGS.num_subset
DATA_DIR = FLAGS.dataroot

fname = os.path.join(DATA_DIR, DATASET_NAME, 'annotations', DATA_SPLIT + '_annotations.json')

with open(fname, 'r') as load_f:
    all_annos = json.load(load_f)

num_data = len(all_annos)
print(num_data)
exit()

### Get a random subset
idx_data = np.arange(num_data, dtype=int)
np.random.shuffle(idx_data)
idx_data = idx_data[:NUM_SUBSET]

### Subset of the data
all_annos_subset = []

for idx in idx_data:
    all_annos_subset.append(all_annos[idx])

print(all_annos_subset)
print(len(all_annos_subset))
print()

print("Writing to file....")
#out_fname = os.path.join(DATA_DIR, DATASET_NAME, 'annotations', DATA_SPLIT + '_annotations_subset.json')

## 15000 training images
#out_fname = os.path.join(DATA_DIR, DATASET_NAME, 'annotations', DATA_SPLIT + '_annotations_bigsubset.json')

out_fname = os.path.join(DATA_DIR, DATASET_NAME, 'annotations', DATA_SPLIT + '_annotations_all.json')

with open(out_fname, 'w') as dump_f:
    json.dump(all_annos_subset, dump_f)

with open(out_fname, 'r') as load_f:
    all_annos_subset = json.load(load_f)

print(all_annos_subset)
print(len(all_annos_subset))
print("Done")
