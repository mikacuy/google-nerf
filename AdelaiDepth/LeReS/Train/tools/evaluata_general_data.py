'''
Mikaela Uy
1021    : Quantatively evaluate scannet scenes to pick the best prior model
        : Also find the scale/shift init for the nerf training 
'''
import math
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
from data.finetune_dataset import FinetuneDataset
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

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_paper_trial2_cropped/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/paper_trial2_cropped', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_scade_rebuttal_reflective_surfaces2/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/AdelaiDepth/LeReS/Train/scade_rebuttal_reflective_surfaces', help='Root dir for dataset')

parser.add_argument('--dump_dir', default= "dump_scade_rebuttal_reflective_surfaces_v2/", type=str)
parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/AdelaiDepth/LeReS/Train/scade_rebuttal_reflective_surfaces_v2', help='Root dir for dataset')


parser.add_argument("--logdir", default="log_0928_all_dataparallel/", help="path to the log directory", type=str)
parser.add_argument("--ckpt", default="epoch56_step0.pth", help="checkpoint", type=str)

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_paper_trial2/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/paper_trial2', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_chair_trial2/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/chair_trial2', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_cardboard_cropped2/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/cardboard_cropped2', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_chair_more/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/chair', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_compressed_misc/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/misc', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_compressed_cardboard_cropped/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/cardboard_cropped', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_compressed_cardboard_v2_2/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/cardboard_v2', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_compressed_cardboard_more/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/cardboard', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_compressed_paper2/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/paper', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_compressed_chair/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/chair', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_compressed_olaf/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild-compressed/olaf', help='Root dir for dataset')


# parser.add_argument('--dump_dir', default= "dump_in_the_wild_basement_alltaskonomy/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild/basement/subset', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_lounge_alltaskonomy/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild/lounge/subset', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_in_the_wild_b_kitchen_alltaskonomy/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/in-the-wild/b_kitchen/subset', help='Root dir for dataset')



# parser.add_argument('--dump_dir', default= "dump_sample_ambiguous_image_alltaskonomy/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/sample_ambiguous_image', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_tanks_and_temples_courtroom_subsample_alltaskonomy/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Courtroom_subsample/train', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_tanks_and_temples_meetingroom_subsample_alltaskonomy/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Meetingroom_subsample/train', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_tanks_and_temples_church_subsample_alltaskonomy/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/coordinate_mvs/processed_scenes/Church_subsample/train', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_tanks_and_temples_courtroom/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/tanks_and_temples/Courtroom/', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_replica_room1/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/Replica_subset/room_1/rgb/', help='Root dir for dataset')

# parser.add_argument('--dump_dir', default= "dump_assembly_sample1/", type=str)
# parser.add_argument('--dataroot', default='/orion/u/mikacuy/assembly_demo/assembly_sample1/', help='Root dir for dataset')


### For the dataset
parser.add_argument('--phase', type=str, default='test', help='Training flag')


parser.add_argument('--backbone', default= "resnext101", type=str)
parser.add_argument('--d_latent', default= 32, type=int)
parser.add_argument('--num_samples', default= 40, type=int)
parser.add_argument('--rescaled', default=False, type=bool)

parser.add_argument('--ada_version', default= "v2", type=str)
parser.add_argument('--cimle_version', default= "enc", type=str)
parser.add_argument('--import_from_logdir', default=False, type=bool)
parser.add_argument('--visu_all', default=False, type=bool)
parser.add_argument('--seed_num', default= 0, type=int)

parser.add_argument('--default_scale', default= 0.5, type=float)
parser.add_argument('--default_shift', default= 0.0, type=float)

parser.add_argument('--is_nsvf', default=False, type=bool)


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
    

    mask = (gt > 0.1)

    if np.sum(mask) == 0 :
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
image_dir = os.path.join(FLAGS.dataroot)
imgs_list = os.listdir(image_dir)
imgs_list.sort()
imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
print(len(imgs_path))

#### Evaluation ######
mini_batch_size = 5
num_sets = int(NUM_SAMPLE/mini_batch_size)
true_num_samples = num_sets*mini_batch_size # just take the floor


with torch.no_grad():
    for i, v in enumerate(imgs_path):

        if ".txt" in v or ".DS_Store" in v:
            continue

        print('processing (%04d)-th image... %s' % (i, v))
        print(v)
        rgb = cv2.imread(v)

        rgb_c = rgb.copy()
        gt_depth = None
        A_resize = cv2.resize(rgb_c, (448, 448))
        rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        img_torch = scale_torch(A_resize)[None, :, :, :]

        data = {}
        data['rgb'] = img_torch
        batch_size = data['rgb'].shape[0]
        C = data['rgb'].shape[1]
        H = data['rgb'].shape[2]
        W = data['rgb'].shape[3]

        ### Repeat for the number of samples
        num_images = data['rgb'].shape[0]
        data['rgb'] = data['rgb'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
        data['rgb'] = data['rgb'].view(-1, C, H, W)

        ### Iterate over the minibatch
        image_fname = []
    
        img_name = "image" + str(i)
        cv2.imwrite(os.path.join(temp_fol, img_name+"-raw.png"), rgb)
        image_fname.append(os.path.join(temp_fol, img_name+"-raw.png"))


        for k in range(num_sets):

            ## Hard coded d_latent
            z = torch.normal(0.0, 1.0, size=(num_images, mini_batch_size, D_LATENT))
            z = z.view(-1, D_LATENT).cuda()

            pred_depth = model.inference(data, z, rescaled=RESCALED)


            # ### Scale the output by mean and sd

            for s in range(mini_batch_size):
                curr_pred_depth = pred_depth[s]


                curr_pred_depth = curr_pred_depth.to("cpu").detach().numpy().squeeze() 

                pred_depth_ori = cv2.resize(curr_pred_depth, (rgb.shape[1], rgb.shape[0]))

                pred_depth_ori = pred_depth_ori[30:, 30:]

                img_name = "image" + str(i) + "_" + str(k) + "_" + str(s)
                

                plt.imsave(os.path.join(temp_fol, img_name+'-depth.png'), pred_depth_ori, cmap='rainbow')                  

                image_fname.append(os.path.join(temp_fol, img_name+'-depth.png'))

                ### Output point cloud
                # f = 577.870605 ### Hardcoded for scannet
                # reconstruct_depth(pred_depth_ori, rgb, pc_fol, img_name, f)
                # reconstruct_depth(curr_pred_depth_scaled, rgb, pc_scaled_fol, img_name, f)


        ### Collate and output to a single image
        height = rgb.shape[0]
        width = rgb.shape[1]   

        #Output to a single image
        new_im = Image.new('RGBA', (width*(1+NUM_SAMPLE), height))

        images = []
        for fname in image_fname:
            images.append(Image.open(fname))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += width

        output_image_filename = os.path.join(DUMP_DIR, str(i) +'_collate.png')
        new_im.save(output_image_filename)      

print("Done.")

























