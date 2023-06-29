'''
Mikaela Uy
mikacuy@stanford.edu
'''
import math
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
from data.finetune_dataset import FinetuneDataset, FinetuneDataset_wild
from lib.models.multi_depth_model_auxiv2 import *
from lib.configs.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt
from tools.parse_arg_base import print_options
from lib.configs.config import cfg, merge_cfg_from_file, print_configs

## for dataloaders
import torch.utils.data

import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *
import torchvision.transforms as transforms

import argparse
from PIL import Image
import random
import imageio
import json


parser = argparse.ArgumentParser()

# parser.add_argument("--logdir", default="log_0926_bigsubset_dataparallel_corrected/", help="path to the log directory", type=str)
# parser.add_argument("--ckpt", default="epoch56_step0.pth", help="checkpoint", type=str)

parser.add_argument("--logdir", default="log_0928_all_dataparallel/", help="path to the log directory", type=str)
parser.add_argument("--ckpt", default="epoch56_step0.pth", help="checkpoint", type=str)

# parser.add_argument('--dump_dir', default= "dump_1108_Auditorium_subsample_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1108_Church_subsample_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1108_Church_subsample2_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1108_Courtroom_subsample_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1108_Courtroom_subsample2_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1108_Meetingroom_subsample_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1108_Meetingroom_subsample_part2_sfmaligned_indv_new/", type=str)

# parser.add_argument('--dump_dir', default= "dump_1109_lounge_v3_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1109_lounge_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1109_b_kitchen_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1109_basement_part2_sfmaligned_indv_new/", type=str)


### For the dataset
parser.add_argument('--phase', type=str, default='test', help='Training flag')

# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Auditorium_subsample/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Church_subsample/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Church_subsample2/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Courtroom_subsample/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Courtroom_subsample2/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Meetingroom_subsample/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Meetingroom_subsample_part2/train/', help='Root dir for dataset')

# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_processed/processed/lounge_v3/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_processed/processed/lounge_v2/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_processed/processed/b_kitchen/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_processed/processed/basement_part2/train/', help='Root dir for dataset')


### In the Wild Corrected --> 0213, for basement and lounge
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_corrected/processed/lounge_corrected_v2/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_corrected/processed/basement_corrected_v2/train/', help='Root dir for dataset')

## Basement two components
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_corrected/processed/basement_corrected_v2_two_components_component1/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_corrected/processed/basement_corrected_v2_two_components_component2/train/', help='Root dir for dataset')

## Lounge is fixed
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_corrected/processed/lounge_corrected/train/', help='Root dir for dataset')
parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes_wild_cleaned0218/lounge_corrected_v2/train/', help='Root dir for dataset')


# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_corrected/processed/basement_corrected/train/', help='Root dir for dataset')

### Fixed basement data
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_corrected/processed/basement_0213/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in_the_wild_corrected/processed/basement_0213_v2/train/', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_0213_lounge_corrected_v2_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_0213_basement_corrected_v2_sfmaligned_indv_new/", type=str)

# parser.add_argument('--dump_dir', default= "dump_0213_basement_corrected_v2_two_components_component1_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_0213_basement_corrected_v2_two_components_component2_sfmaligned_indv_new/", type=str)

parser.add_argument('--dump_dir', default= "dump_0213_lounge_corrected_v2_sfmaligned_indv_new_visu/", type=str)

# parser.add_argument('--dump_dir', default= "dump_0213_basement_corrected_sfmaligned_indv_new/", type=str)

### Fixed basement data
# parser.add_argument('--dump_dir', default= "dump_0213_basement_0213_sfmaligned_indv_new/", type=str)
# parser.add_argument('--dump_dir', default= "dump_0213_basement_0213_v2_sfmaligned_indv_new/", type=str)


#############################


### Sparsity experiment
# parser.add_argument('--dump_dir', default= "dump_0128_scene0710_7_sfmaligned_indv/", type=str)
# parser.add_argument('--dump_dir', default= "dump_0128_scene0758_1_sfmaligned_indv/", type=str)
# parser.add_argument('--dump_dir', default= "dump_0128_scene0781_7_sfmaligned_indv/", type=str)

# parser.add_argument('--dump_dir', default= "dump_dummy/", type=str)


# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/1/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/2/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/3/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/4/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/5/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/6/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/7/train/', help='Root dir for dataset')

# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/1/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/2/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/3/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/4/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/5/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/6/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/7/train/', help='Root dir for dataset')

# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/1/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/2/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/3/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/4/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/5/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/6/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/7/train/', help='Root dir for dataset')
###########################


parser.add_argument('--backbone', default= "resnext101", type=str)
parser.add_argument('--d_latent', default= 32, type=int)
parser.add_argument('--num_samples', default= 20, type=int)
parser.add_argument('--rescaled', default=False, type=bool)

parser.add_argument('--ada_version', default= "v2", type=str)
parser.add_argument('--cimle_version', default= "enc", type=str)
parser.add_argument('--import_from_logdir', default=False, type=bool)
parser.add_argument('--visu_all', default=True, type=bool)
parser.add_argument('--seed_num', default= 0, type=int)

parser.add_argument('--default_scale', default= 0.5, type=float)
parser.add_argument('--default_shift', default= 0.0, type=float)

parser.add_argument('--is_nsvf', default=False, type=bool)
parser.add_argument('--is_wild', default=False, type=bool)


FLAGS = parser.parse_args()
LOG_DIR = FLAGS.logdir
CKPT = FLAGS.ckpt

IMPORT_FROM_LOGDIR = FLAGS.import_from_logdir
VISU_ALL = FLAGS.visu_all
SEED_NUM = FLAGS.seed_num
torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)
random.seed(SEED_NUM)

IS_NSVF = FLAGS.is_nsvf
IS_WILD = FLAGS.is_wild

#### Import from LOG_DIR files or from global files
if IMPORT_FROM_LOGDIR:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, "../", LOG_DIR))
    from data.load_dataset_distributed import MultipleDataLoaderDistributed, MultipleDatasetDistributed
    from lib.models.multi_depth_model_auxiv2 import *
    from lib.configs.config import cfg, merge_cfg_from_file, print_configs
    from lib.utils.net_tools import save_ckpt, load_ckpt
    from tools.parse_arg_base import print_options
    from lib.configs.config import cfg, merge_cfg_from_file, print_configs
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, "../"))
    from data.load_dataset_distributed import MultipleDataLoaderDistributed, MultipleDatasetDistributed
    from lib.models.multi_depth_model_auxiv2 import *
    from lib.configs.config import cfg, merge_cfg_from_file, print_configs
    from lib.utils.net_tools import save_ckpt, load_ckpt
    from tools.parse_arg_base import print_options
    from lib.configs.config import cfg, merge_cfg_from_file, print_configs

ADA_VERSION = FLAGS.ada_version
CIMLE_VERSION = FLAGS.cimle_version
print(CIMLE_VERSION)
print(ADA_VERSION)
print("===================")

DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
temp_fol = os.path.join(DUMP_DIR, "tmp")
if not os.path.exists(temp_fol): os.mkdir(temp_fol)
pc_fol = os.path.join(DUMP_DIR, "pc")
if not os.path.exists(pc_fol): os.mkdir(pc_fol)
gt_fol = os.path.join(DUMP_DIR, "gt")
if not os.path.exists(gt_fol): os.mkdir(gt_fol)

LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

RESCALED = FLAGS.rescaled
D_LATENT = FLAGS.d_latent
NUM_SAMPLE = FLAGS.num_samples

print("Rescaled: "+str(RESCALED))

### Merge config with current configs
merge_cfg_from_file(FLAGS)

if CIMLE_VERSION == "enc":
    model = RelDepthModel_cIMLE(d_latent=D_LATENT, version=ADA_VERSION)
else:
    model = RelDepthModel_cIMLE_decoder(d_latent=D_LATENT, version=ADA_VERSION)

model.cuda()

### Load model
model_dict = model.state_dict()

# CKPT_FILE = os.path.join("outputs", LOG_DIR, "ckpt", CKPT)
CKPT_FILE = os.path.join(LOG_DIR, "ckpt", CKPT)

if os.path.isfile(CKPT_FILE):
    print("loading checkpoint %s" % CKPT_FILE)
    checkpoint = torch.load(CKPT_FILE)

    checkpoint['model_state_dict'] = strip_prefix_if_present(checkpoint['model_state_dict'], "module.")
    depth_keys = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict} ## <--- some missing keys in the loaded model from the given model
    print(len(depth_keys))

    # Overwrite entries in the existing state dict
    model_dict.update(depth_keys)        

    # Load the new state dict
    model.load_state_dict(model_dict)


    print("Model loaded.")

else:
	print("Error: Model does not exist.")
	exit()

mean0, var0, mean1, var1, mean2, var2, mean3, var3 = load_mean_var_adain(os.path.join(LOG_DIR, "mean_var_adain.npy"), torch.device("cuda"))
model.set_mean_var_shifts(mean0, var0, mean1, var1, mean2, var2, mean3, var3)
print("Initialized adain mean and var.")

### Quantitative Metrics ####
def evaluate_rel_err(pred, gt, mask_invalid=None, scale=10.0 ):
    if type(pred).__module__ != np.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.cpu().numpy()

    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    if pred.shape != gt.shape:
        logger.info('The shapes of dt and gt are not same!')
        return -1
    if mask_invalid is not None:
        gt = gt[~mask_invalid]
        pred = pred[~mask_invalid]


    mask = (gt > 1e-8)
    gt = gt[mask]
    pred = pred[mask]
    n_pxl = gt.size
    gt_scale = gt * scale
    pred_scale = pred * scale

    err_absRel = -1.
    err_squaRel = -1.
    err_silog = -1.
    err_delta1 = -1.
    err_whdr = -1.


    if gt_scale.size < 10:
        print('Valid pixel size:', gt_scale.size, 'Invalid evaluation!!!!')
        exit()
        return err_absRel, err_squaRel, err_silog, err_delta1, err_whdr

    #Mean Absolute Relative Error
    rel = np.abs(gt - pred) / gt# compute errors
    abs_rel_sum = np.sum(rel)
    err_absRel = np.float64(abs_rel_sum) / float(n_pxl)

    #Square Mean Relative Error
    s_rel = ((gt_scale - pred_scale) * (gt_scale - pred_scale)) / (gt_scale * gt_scale)# compute errors
    squa_rel_sum = np.sum(s_rel)
    err_squaRel = np.float64(squa_rel_sum) / float(n_pxl)

    # Scale invariant error
    diff_log = np.log(pred_scale) - np.log(gt_scale)
    diff_log_sum = np.sum(diff_log)
    err_silog = np.float64(diff_log_sum)/ float(n_pxl)

    #Delta
    gt_pred = gt_scale / pred_scale
    pred_gt = pred_scale / gt_scale
    gt_pred = np.reshape(gt_pred, (1, -1))
    pred_gt = np.reshape(pred_gt, (1, -1))
    gt_pred_gt = np.concatenate((gt_pred, pred_gt), axis=0)
    ratio_max = np.amax(gt_pred_gt, axis=0)

    delta_1_sum = np.sum(ratio_max < 1.25)
    err_delta1 = np.float64(delta_1_sum)/ float(n_pxl)

    # WHDR error
    whdr_err_sum, eval_num = weighted_human_disagreement_rate(gt_scale, pred_scale)
    err_whdr = np.float64(whdr_err_sum)/ float(eval_num)

    return err_absRel, err_squaRel, err_silog, err_delta1, err_whdr


def weighted_human_disagreement_rate(gt, pred):
    p12_index = select_index(gt)
    gt_reshape = np.reshape(gt, gt.size)
    pred_reshape = np.reshape(pred, pred.size)
    mask = gt > 0
    gt_p1 = gt_reshape[mask][p12_index['p1']]
    gt_p2 = gt_reshape[mask][p12_index['p2']]
    pred_p1 = pred_reshape[mask][p12_index['p1']]
    pred_p2 = pred_reshape[mask][p12_index['p2']]

    p12_rank_gt = np.zeros_like(gt_p1)
    p12_rank_gt[gt_p1 > gt_p2] = 1
    p12_rank_gt[gt_p1 < gt_p2] = -1

    p12_rank_pred = np.zeros_like(gt_p1)
    p12_rank_pred[pred_p1 > pred_p2] = 1
    p12_rank_pred[pred_p1 < pred_p2] = -1

    err = np.sum(p12_rank_gt != p12_rank_pred)
    valid_pixels = gt_p1.size
    return err, valid_pixels


def select_index(gt_depth, select_size=10000):
    valid_size = np.sum(gt_depth>0)
    try:
        p = np.random.choice(valid_size, select_size*2, replace=False)
    except:
        p = np.random.choice(valid_size, select_size*2*2, replace=True)
    np.random.shuffle(p)
    p1 = p[0:select_size*2:2]
    p2 = p[1:select_size*2:2]

    p12_index = {'p1': p1, 'p2': p2}
    return p12_index


### Get valid gt depths 
def transform_shift_scale(depth, valid_threshold=-1e-8, max_threshold=1e8):
    mask = (depth > valid_threshold) & (depth < max_threshold)     
    gt_maskbatch = depth[mask]

    # Get mean and standard deviation
    data_mean = []
    data_std_dev = []
    for i in range(depth.shape[0]):
        gt_i = depth[i]
        mask = gt_i > 0
        depth_valid = gt_i[mask]
        depth_valid = depth_valid[:5]
        if depth_valid.shape[0] < 10:
            data_mean.append(torch.tensor(0).cuda())
            data_std_dev.append(torch.tensor(1).cuda())
            continue
        size = depth_valid.shape[0]
        depth_valid_sort, _ = torch.sort(depth_valid, 0)
        depth_valid_mask = depth_valid_sort[int(size*0.1): -int(size*0.1)]
        data_mean.append(depth_valid_mask.mean())
        data_std_dev.append(depth_valid_mask.std())
    data_mean = torch.stack(data_mean, dim=0).cuda()
    data_std_dev = torch.stack(data_std_dev, dim=0).cuda()

    gt_trans = (depth - data_mean[:, None, None, None]) / (data_std_dev[:, None, None, None] + 1e-8)
    
    return gt_trans

### Tranform pred to fit the ground truth

#### Fit with regularization ####
def recover_metric_depth(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    

    mask = (gt > 0.5)

    if np.sum(mask) == 0 :
        print("Warning: No SfM points found.")
        return pred, FLAGS.default_scale, FLAGS.default_shift

    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    
    pred_metric = a * pred + b

    # pred_metric[~mask] = 0.

    return pred_metric, a, b

#### Image transform #####
def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

def remap_color_to_depth(depth_img):
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_TURBO).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))

    depth = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, depth_img)
    return depth    
##############################

### Compute RSME ###
def compute_rmse(prediction, target):
    return np.sqrt(np.mean(np.square(prediction - target)))

def per_pixel_error(prediction, target):
    return np.sqrt(np.square(prediction - target))

### Dataset

### Dataloader
# datapath = os.path.join(FLAGS.dataroot, FLAGS.scenename)
datapath = FLAGS.dataroot

if not IS_WILD:
    dataset = FinetuneDataset(datapath, "processed", is_nsvf=IS_NSVF, split="test", data_aug=False)
else:
    dataset = FinetuneDataset_wild(datapath, "processed", is_nsvf=IS_NSVF, split="test", data_aug=False)


### Create output dir for the multiple hypothesis
hypothesis_outdir = os.path.join(FLAGS.dataroot, "leres_cimle", DUMP_DIR)
if not os.path.exists(hypothesis_outdir): os.makedirs(hypothesis_outdir)

##### Also load intrinsics and depth scale for the dataset. #####
json_fname =  os.path.join(datapath, '../transforms_train.json')
with open(json_fname, 'r') as fp:
    meta = json.load(fp)

depth_scaling_factor = float(meta['depth_scaling_factor'])
#################################################################

scaleshift_outdir = os.path.join(FLAGS.dataroot, "scale_shift_inits", DUMP_DIR)
if not os.path.exists(scaleshift_outdir): os.makedirs(scaleshift_outdir)


#### Evaluation ######
zcache_dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False)
print(len(zcache_dataloader))

mini_batch_size = 5
num_sets = int(NUM_SAMPLE/mini_batch_size)
true_num_samples = num_sets*mini_batch_size # just take the floor


### For quantitative evaluation
total_gt_rsme = 0.0
total_sfm_rsme = 0.0
num_evaluated = 0

model.eval()

### Focal length for scannet
if not IS_NSVF:
    f = 577.870605 ### this focal length is not exactly right for DDP images, but it doesn't matter here
else:
    ### Check this for other models
    f = 1111.111

with torch.no_grad():

    all_scales = []
    all_shifts = []

    best_sfm_scales = []
    best_sfm_shifts = []

    all_npercentile = []
    all_tpercentile = []

    for i, data in enumerate(zcache_dataloader):

        batch_size = data['rgb'].shape[0]
        C = data['rgb'].shape[1]
        H = data['rgb'].shape[2]
        W = data['rgb'].shape[3]

        ### Repeat for the number of samples
        num_images = data['rgb'].shape[0]
        data['rgb'] = data['rgb'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
        data['rgb'] = data['rgb'].view(-1, C, H, W)


        rgb = torch.clone(data['rgb'][0]).permute(1, 2, 0).to("cpu").detach().numpy() 
        rgb = rgb[:, :, ::-1] ## dataloader is bgr
        rgb = 255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb = np.array(rgb, np.int)


        ### Raw gt depth ###
        curr_depth_path = data['B_paths'][0]

        depth_img = cv2.imread(curr_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        valid_depth = depth_img > 0.5

        if not IS_NSVF:
            ## Scannet depth
            depth_img = (depth_img/depth_scaling_factor).astype(np.float32)
        else:
            depth_img = remap_color_to_depth(depth_img)
            depth_img = depth_img.astype(float)

        orig_shape = depth_img.shape

        ### For computing scale and shift
        depth_orig_size = depth_img.copy()
        depth_img = cv2.resize(depth_img, (448, 448), interpolation=cv2.INTER_NEAREST)

        #### Get image focal length
        frame = meta['frames'][i]
        fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
        intrinsics = np.array((fx, fy, cx, cy))
        ###########################        

        ### Load sparse SfM depth
        curr_sfm_depth_path = data['C_paths'][0]
        sfm_depth_img = cv2.imread(curr_sfm_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
        sfm_depth_img = (sfm_depth_img/depth_scaling_factor).astype(np.float32)

        ### Filter out SfM points that are beyond the far plane
        sfm_depth_img[sfm_depth_img > meta["far"]] = 0.0
        valid_sfm_depth = sfm_depth_img > 0.5


        ### Iterate over the minibatch
        image_fname = []

        if i%10==0  or VISU_ALL:      
            img_name = "image" + str(i)
            cv2.imwrite(os.path.join(temp_fol, img_name+"-raw.png"), rgb)
            image_fname.append(os.path.join(temp_fol, img_name+"-raw.png"))
            img_name = "image" + str(i) + "_gt"
            reconstruct_depth_intrinsics(depth_img, rgb, gt_fol, img_name, intrinsics)


        all_gt_rsme = np.zeros((batch_size, mini_batch_size*num_sets))
        all_sfm_rsme = np.zeros((batch_size, mini_batch_size*num_sets))

        gt_image_scales = []
        gt_image_shifts = []

        sfm_image_scales = []
        sfm_image_shifts = []

        all_pred_depths = []
        all_pp_error = []

        all_npercentile_error = []
        all_tpercentile_error = []

        for k in range(num_sets):

            ## Hard coded d_latent
            z = torch.normal(0.0, 1.0, size=(num_images, mini_batch_size, D_LATENT))
            z = z.view(-1, D_LATENT).cuda()

            pred_depth = model.inference(data, z, rescaled=RESCALED)

            for s in range(mini_batch_size):
                curr_pred_depth = pred_depth[s]
                curr_pred_depth = curr_pred_depth.to("cpu").detach().numpy().squeeze() 

                pred_depth_ori = curr_pred_depth
                # pred_depth_ori = cv2.resize(curr_pred_depth, (H, W))

                img_name = "image" + str(i) + "_" + str(k) + "_" + str(s)
                

                ### Resize first then compute for error
                curr_pred_depth_raw = cv2.resize(curr_pred_depth, (depth_orig_size.shape[1], depth_orig_size.shape[0])) ### check this resize function 

                ### Align prediction and compute qualitative results
                ## Raw depth
                curr_pred_depth_metric, curr_scale, curr_shift = recover_metric_depth(curr_pred_depth_raw, depth_orig_size)

                gt_image_scales.append(curr_scale)
                gt_image_shifts.append(curr_shift)

                gt_depth_rmse = compute_rmse(curr_pred_depth_metric[valid_depth], depth_orig_size[valid_depth])

                pixelwise_error = per_pixel_error(curr_pred_depth_metric[valid_depth], depth_orig_size[valid_depth])
                n_error = np.percentile(pixelwise_error, 90)
                t_error = np.percentile(pixelwise_error, 80)
                all_npercentile_error.append(n_error)
                all_tpercentile_error.append(t_error)


                ## SfM depth
                curr_pred_sfm_depth_metric, curr_sfm_scale, curr_sfm_shift = recover_metric_depth(curr_pred_depth_raw, sfm_depth_img)

                sfm_image_scales.append(curr_sfm_scale)
                sfm_image_shifts.append(curr_sfm_shift)

                sfm_depth_rmse = compute_rmse(curr_pred_sfm_depth_metric[valid_depth], depth_orig_size[valid_depth])                

                curr_pp_error = per_pixel_error(curr_pred_sfm_depth_metric, depth_orig_size)

                if i%10==0  or VISU_ALL:
                    # save depth
                    plt.imsave(os.path.join(temp_fol, img_name+'-depth.png'), curr_pred_sfm_depth_metric, cmap='rainbow')
                    image_fname.append(os.path.join(temp_fol, img_name+'-depth.png'))

                    ### Output point cloud
                    reconstruct_depth_intrinsics(curr_pred_depth, rgb, pc_fol, img_name, intrinsics)

                    rgb_orig = cv2.imread(data['A_paths'][0])
                    reconstruct_depth_intrinsics(curr_pred_sfm_depth_metric, rgb_orig, pc_fol, img_name+"-sfmscaled", intrinsics)
                    reconstruct_depth_intrinsics(curr_pred_depth_metric, rgb_orig, pc_fol, img_name+"-gtscaled", intrinsics)


                ##### Save the current RSME
                all_gt_rsme[:, k*mini_batch_size + s] = gt_depth_rmse
                all_sfm_rsme[:, k*mini_batch_size + s] = sfm_depth_rmse


                ### Change to using aligned depth instead
                all_pred_depths.append(curr_pred_sfm_depth_metric)
                all_pp_error.append(curr_pp_error)

            #######

        #### this is actually per pixel alignment error


        all_pred_depths = np.stack(all_pred_depths)
        all_pp_error = np.stack(all_pp_error)

        # # if np.sum(valid_sfm_depth) > 0:
        # best_per_pixel = np.min(all_pp_error, axis=0)
        # best_per_pixel[~valid_depth] = 0



        # plt.clf()
        # # plt.rcParams['font.size'] = '4'

        # fig, ax = plt.subplots()

        # plt.title('Num SfM points: '+str(np.sum(valid_sfm_depth))+ '\nper-pixel min error - post alignement')
        # im = ax.imshow(best_per_pixel, cmap='rainbow')
        # plt.colorbar(im, ax=ax)

        # plt.axis("off")

        # # plt.tight_layout()
        # plt.savefig(os.path.join(DUMP_DIR, "Image_"+str(i)))

        # print(all_pred_depths.shape)
        # print(best_per_pixel.shape)
        # exit()
        
        ### Get the best SfM scale and append this
        sfm_idx_to_take = np.argmin(all_sfm_rsme, axis=-1)[0]
        best_sfm_scales.append(sfm_image_scales[sfm_idx_to_take])
        best_sfm_shifts.append(sfm_image_shifts[sfm_idx_to_take])
        total_sfm_rsme += all_sfm_rsme[0][sfm_idx_to_take]

        ### Gt mean scales and shifts
        gt_image_scales = np.array(gt_image_scales)
        gt_image_shifts = np.array(gt_image_shifts)
        all_scales.append(np.mean(gt_image_scales))
        all_shifts.append(np.mean(gt_image_shifts)) 

        idx_to_take = np.argmin(all_gt_rsme, axis=-1)[0]
        total_gt_rsme += all_gt_rsme[0][idx_to_take]
        num_evaluated += 1       

        # ### 20th and 80th percentile
        # best_n_percentile = np.array(all_npercentile_error)[idx_to_take]
        # best_t_percentile = np.array(all_tpercentile_error)[idx_to_take]
        # all_npercentile.append(best_n_percentile)
        # all_tpercentile.append(best_t_percentile)

        
        # ### Save scale/shift init for the image
        # #### Save into numpy array in the dump dir
        # curr_rbg_name = data['A_paths'][0]
        # # print(curr_rbg_name)

        # fname = curr_rbg_name.split("/")[-1][:-4] + "_sfminit.npy"

        # curr_scaleshift = np.array([sfm_image_scales[sfm_idx_to_take], sfm_image_shifts[sfm_idx_to_take]])
        # outfname = os.path.join(scaleshift_outdir, fname)
        # np.save(outfname, curr_scaleshift)

        # scaleshift = np.load(outfname).astype(np.float64)
        # # print(scaleshift)
        # # print(scaleshift.shape)
        # # print(outfname)
        # # print()

        # fname = curr_rbg_name.split("/")[-1][:-4] + "_gtinit.npy"

        # curr_scaleshift = np.array([gt_image_scales[idx_to_take], gt_image_shifts[idx_to_take]])
        # outfname = os.path.join(scaleshift_outdir, fname)
        # np.save(outfname, curr_scaleshift)

        # scaleshift = np.load(outfname).astype(np.float64)
        # print(scaleshift)
        # print(scaleshift.shape)
        # print(outfname)
        # print()

        # ###########################
        # if np.sum(valid_sfm_depth) > 0:
        #     idx_selected = sfm_idx_to_take
        #     # idx_selected = idx_to_take

        #     print(idx_selected)
        #     ## Output in matplotlib
        #     plt.clf()
        #     plt.rcParams['font.size'] = '4'

        #     fig, axs = plt.subplots(4, 2)
        #     fig.suptitle('Num SfM points: '+str(np.sum(valid_sfm_depth)))

        #     ### SfM points
        #     axs[0, 0].scatter(all_pred_depths[idx_selected][valid_depth], depth_orig_size[valid_depth], s=0.2, c="g")
        #     axs[0, 0].scatter(all_pred_depths[idx_selected][valid_sfm_depth], sfm_depth_img[valid_sfm_depth], s=4.0, c="b")

        #     # axs[0].set_title('Num SfM points: '+str(np.sum(valid_sfm_depth)))
        #     axs[0, 0].set(xlabel='Best hypothesis prediction')
        #     axs[0, 0].set(ylabel='SfM/GT points')


        #     ### Draw line that 
        #     axs[0, 0].axline((0.0, sfm_image_shifts[idx_selected]), (1.0, sfm_image_scales[idx_selected]+sfm_image_shifts[idx_selected]), c='b')
        #     axs[0, 0].axline((0.0, gt_image_shifts[idx_selected]), (1.0, gt_image_scales[idx_selected]+gt_image_shifts[idx_selected]), c='r')
        #     axs[0, 0].set_xlim(np.min(all_pred_depths[idx_selected][valid_depth])-0.01, np.max(all_pred_depths[idx_selected][valid_depth])+0.01)
        #     axs[0, 0].set_ylim(np.min(depth_orig_size[valid_depth])-0.01, np.max(depth_orig_size[valid_depth])+0.01)


        #     ### SfM points
        #     axs[0, 1].scatter(all_pred_depths[idx_selected][valid_sfm_depth], sfm_depth_img[valid_sfm_depth], s=4.0, c="b")
        #     axs[0, 1].set(xlabel='Best hypothesis prediction')
        #     # axs[1].set(ylabel='SfM points')


        #     ### Draw line that 
        #     axs[0, 1].axline((0.0, sfm_image_shifts[idx_selected]), (1.0, sfm_image_scales[idx_selected]+sfm_image_shifts[idx_selected]), c='b')
        #     axs[0, 1].set_xlim(np.min(all_pred_depths[idx_selected][valid_sfm_depth])-0.01, np.max(all_pred_depths[idx_selected][valid_sfm_depth])+0.01)
        #     axs[0, 1].set_ylim(np.min(sfm_depth_img[valid_sfm_depth])-0.01, np.max(sfm_depth_img[valid_sfm_depth])+0.01)


        #     #### Plot GT depth map and LeReS raw output
        #     axs[1, 0].set_title("GT depth map normalized")
        #     im3 = axs[1, 0].imshow(depth_orig_size, cmap='rainbow', vmin=0, vmax=10)
        #     axs[1, 0].get_xaxis().set_visible(False)
        #     axs[1, 0].get_yaxis().set_visible(False)
        #     plt.colorbar(im3, ax=axs[1, 0])

        #     axs[1, 1].set_title("Raw LeReS output normalized")
        #     im4 = axs[1, 1].imshow(all_pred_depths[idx_selected], cmap='rainbow', vmin=0, vmax=10)
        #     axs[1, 1].get_xaxis().set_visible(False)
        #     axs[1, 1].get_yaxis().set_visible(False)
        #     plt.colorbar(im4, ax=axs[1, 1])

        #     #### Corrected depth
        #     corrected_depth_gt = all_pred_depths[idx_selected]*gt_image_scales[idx_selected] + gt_image_shifts[idx_selected]
        #     corrected_depth_sfm = all_pred_depths[idx_selected]*sfm_image_scales[idx_selected] + sfm_image_shifts[idx_selected]


        #     #### Plot GT depth map and LeReS raw output
        #     axs[2, 0].set_title("Aligned with GT")
        #     im5 = axs[2, 0].imshow(corrected_depth_gt, cmap='rainbow', vmin=0, vmax=10)
        #     axs[2, 0].get_xaxis().set_visible(False)
        #     axs[2, 0].get_yaxis().set_visible(False)
        #     plt.colorbar(im5, ax=axs[2, 0])

        #     axs[2, 1].set_title("Aligned with SfM")
        #     im6 = axs[2, 1].imshow(corrected_depth_sfm, cmap='rainbow', vmin=0, vmax=10)
        #     axs[2, 1].get_xaxis().set_visible(False)
        #     axs[2, 1].get_yaxis().set_visible(False)
        #     plt.colorbar(im6, ax=axs[2, 1])


            
        #     corrected_depth_sfm[~valid_depth] = 0
        #     corrected_depth_sfm[~valid_depth] = 0

        #     error_gt = per_pixel_error(corrected_depth_gt, depth_orig_size)
        #     error_sfm = per_pixel_error(corrected_depth_sfm, depth_orig_size)

        #     axs[3, 0].set_title("Residual best alignment scale/shift with GT")
        #     im1 = axs[3, 0].imshow(error_gt, cmap='rainbow')
        #     plt.colorbar(im1, ax=axs[3, 0])

        #     axs[3, 0].get_xaxis().set_visible(False)
        #     axs[3, 0].get_yaxis().set_visible(False)

        #     axs[3, 1].set_title("Residual best alignment scale/shift with SfM points")
        #     im2 = axs[3, 1].imshow(error_sfm, cmap='rainbow')
        #     plt.colorbar(im2, ax=axs[3, 1])

        #     axs[3, 1].get_xaxis().set_visible(False)
        #     axs[3, 1].get_yaxis().set_visible(False)

        #     # plt.tight_layout()
        #     plt.savefig(os.path.join(DUMP_DIR, "Image_"+str(i)), dpi = 200)
            # exit()

        ##########################


        # #### Scaled output
        # for k in range(num_sets):
        #     for s in range(mini_batch_size):
        #         curr_depth = all_pred_depths[k*mini_batch_size+s]

        #         img_name = "image" + str(i) + "_" + str(k) + "_" + str(s)
        #         if i%10==0  or VISU_ALL:
        #             scaled_depth = curr_depth*curr_scaleshift[0] + curr_scaleshift[1]
        #             rgb_resized = cv2.resize(rgb, (scaled_depth.shape[1], scaled_depth.shape[0]))
        #             reconstruct_depth_intrinsics(scaled_depth, rgb_resized, pc_fol, img_name+"-sfmunifiedscaled", intrinsics)


        print("===============")
        print("i")
        ### Output to file
        for k in range(num_sets):
            for s in range(mini_batch_size):

                curr_depth = all_pred_depths[k*mini_batch_size+s]

                curr_rbg_name = data['A_paths'][0]

                if not IS_WILD:
                    fname = curr_rbg_name.split("/")[-1][:-4] + "_" + str(k*mini_batch_size+s) + ".npy"
                else:
                    fname = curr_rbg_name.split("/")[-1][:-5] + "_" + str(k*mini_batch_size+s) + ".npy"


                outfname = os.path.join(hypothesis_outdir, fname)

                # ### Debug
                # print(depth_orig_size.shape)
                # print(curr_depth.shape)
                # exit()

                np.save(outfname, np.array(curr_depth))                

                ## Check output --> debug
                depth = np.load(outfname).astype(np.float64)
                print(depth)
                print(depth.shape)
                print(outfname)
                print()


        if i%10==0  or VISU_ALL:
            ### Collate and output to a single image
            height = H
            width = W    

            #Output to a single image
            for k in range(num_sets):

                new_im = Image.new('RGBA', (width*(1+mini_batch_size), height))

                curr_image_fname = []
                curr_image_fname.append(image_fname[0])
                for j in range(k*mini_batch_size, (k+1)*mini_batch_size):
                    curr_image_fname.append(image_fname[j])

                images = []

                for fname in curr_image_fname:
                    images.append(Image.open(fname))

                x_offset = 0
                for im in images:
                    new_im.paste(im, (x_offset,0))
                    x_offset += width

                output_image_filename = os.path.join(DUMP_DIR, str(i) + "_" + str(k) + '_collate.png')
                new_im.save(output_image_filename) 
        # print(output_image_filename)

        if i%100==0:
            print("Finished "+str(i)+"/"+str(len(zcache_dataloader)
                )+".")

mean_gt_rsme = total_gt_rsme/float(num_evaluated)
mean_sfm_rsme = total_sfm_rsme/float(num_evaluated)


log_string("=" * 20)
log_string("")
log_string("Num evaluated= "+str(num_evaluated))
log_string("")
log_string("Mean GT RSME= "+str(mean_gt_rsme))
log_string("")             
log_string("Mean SfM RSME = "+str(mean_sfm_rsme))
log_string("")           

print("Done.")



print("Scales:")
print(all_scales)
print()
print("Shifts:")
print(all_shifts)

print()
print("Best SfM Scales:")
print(best_sfm_scales)
print()
print("Best SfM Shifts:")
print(best_sfm_shifts)
print()
print()


# print("90th percentile error")
# print(all_npercentile)
# print(np.median(np.array(all_npercentile)))
# print(np.min(np.array(all_npercentile)))
# print(np.mean(np.array(all_npercentile)))
# print("========")
# print()

# print("80th percentile error")
# print(all_tpercentile)
# print(np.median(np.array(all_tpercentile)))
# print(np.min(np.array(all_tpercentile)))
# print(np.mean(np.array(all_tpercentile)))






