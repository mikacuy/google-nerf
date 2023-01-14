'''
Mikaela Uy
Fixing/debugging metric and focal
'''
import math
import os, sys

## for dataloaders
import torch.utils.data

import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import *

import argparse
from PIL import Image
import random

### For 3D reconstruction
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))

from lib.test_utils import refine_focal, refine_shift
from lib.spvcnn_classsification import SPVCNN_CLASSIFICATION

from scipy.interpolate import griddata


parser = argparse.ArgumentParser()

### Currently the best model
# parser.add_argument("--logdir", default="log_0726_lrfixed_001/", help="path to the log directory", type=str)
# parser.add_argument("--ckpt", default="epoch104_step39375.pth", help="checkpoint", type=str)

parser.add_argument("--logdir", default="log_0926_bigsubset_dataparallel_corrected/", help="path to the log directory", type=str)
parser.add_argument("--ckpt", default="epoch56_step0.pth", help="checkpoint", type=str)

# parser.add_argument("--logdir", default="log_0823_encv2_noaug_run2/", help="path to the log directory", type=str)
# parser.add_argument("--ckpt", default="epoch64_step0.pth", help="checkpoint", type=str)
# parser.add_argument('--dump_dir', default= "dump_1027_visutaskonomy_metric/", type=str)
# parser.add_argument('--dump_dir', default= "dump_1027_visutaskonomy_withrecons_backproject2/", type=str)

parser.add_argument('--dump_dir', default= "dump_1029_backproject_debug/", type=str)

### For the dataset
parser.add_argument('--phase', type=str, default='test', help='Training flag')
parser.add_argument('--phase_anno', type=str, default='test', help='Annotations file name')
parser.add_argument('--dataset_list', default=["taskonomy"], nargs='+', help='The names of multiple datasets')
parser.add_argument('--dataset', default='multi', help='Dataset loader name')
parser.add_argument('--dataroot', default='/orion/downloads/coordinate_mvs/', help='Root dir for dataset')

parser.add_argument('--backbone', default= "resnext101", type=str)
parser.add_argument('--d_latent', default= 32, type=int)
parser.add_argument('--num_samples', default= 5, type=int)
parser.add_argument('--rescaled', default=True, type=bool)

parser.add_argument('--ada_version', default= "v2", type=str)
parser.add_argument('--cimle_version', default= "enc", type=str)
parser.add_argument('--import_from_logdir', default=False, type=bool)
parser.add_argument('--visu_all', default=False, type=bool)
parser.add_argument('--seed_num', default= 0, type=int)

### Pretrained LeReS model --> get focal and shift models
parser.add_argument("--leres_pretrained", default="../Minist_Test/res101.pth", help="checkpoint", type=str)

FLAGS = parser.parse_args()
LOG_DIR = FLAGS.logdir
CKPT = FLAGS.ckpt
IMPORT_FROM_LOGDIR = FLAGS.import_from_logdir
VISU_ALL = FLAGS.visu_all

SEED_NUM = FLAGS.seed_num
torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)
random.seed(SEED_NUM)

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
pc_scaled_fol = os.path.join(DUMP_DIR, "pc_scaled")
if not os.path.exists(pc_scaled_fol): os.mkdir(pc_scaled_fol)
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


######################################################
### For scale and focal models from pretrained LeReS
######################################################
def make_shift_focallength_models():
    shift_model = SPVCNN_CLASSIFICATION(input_channel=3,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    focal_model = SPVCNN_CLASSIFICATION(input_channel=5,
                                        num_classes=1,
                                        cr=1.0,
                                        pres=0.01,
                                        vres=0.01
                                        )
    shift_model.eval()
    focal_model.eval()
    return shift_model, focal_model

def reconstruct3D_from_depth(rgb, pred_depth, shift_model, focal_model):
    cam_u0 = rgb.shape[1] / 2.0
    cam_v0 = rgb.shape[0] / 2.0
    pred_depth_norm = pred_depth - pred_depth.min() + 0.5

    dmax = np.percentile(pred_depth_norm, 98)
    pred_depth_norm = pred_depth_norm / dmax

    # proposed focal length, FOV is 60', Note that 60~80' are acceptable.
    proposed_scaled_focal = (rgb.shape[0] // 2 / np.tan((60/2.0)*np.pi/180))

    # recover focal
    focal_scale_1 = refine_focal(pred_depth_norm, proposed_scaled_focal, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_1 = proposed_scaled_focal / focal_scale_1.item()

    # recover shift
    shift_1 = refine_shift(pred_depth_norm, shift_model, predicted_focal_1, cam_u0, cam_v0)
    shift_1 = shift_1 if shift_1.item() < 0.6 else torch.tensor([0.6])
    depth_scale_1 = pred_depth_norm - shift_1.item()

    # recover focal
    focal_scale_2 = refine_focal(depth_scale_1, predicted_focal_1, focal_model, u0=cam_u0, v0=cam_v0)
    predicted_focal_2 = predicted_focal_1 / focal_scale_2.item()

    return shift_1, predicted_focal_2, depth_scale_1


### Pretrained LeReS model on monocular depth estimation ###
LERES_CKPT_FILE = FLAGS.leres_pretrained
print("Loading pretrained LeReS model " + LERES_CKPT_FILE)
checkpoint = torch.load(LERES_CKPT_FILE)

shift_model, focal_model = make_shift_focallength_models()
shift_model.load_state_dict(strip_prefix_if_present(checkpoint['shift_model'], 'module.'),
                                    strict=True)
focal_model.load_state_dict(strip_prefix_if_present(checkpoint['focal_model'], 'module.'),
                                    strict=True)

shift_model.cuda()
focal_model.cuda()
print("Shift and Focal model loaded.")
######################################################


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
def recover_metric_depth(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) & (pred > 1e-8)

    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    pred_metric = a * pred + b
    return pred_metric

##############################


### Dataset
dataset = MultipleDatasetDistributed(FLAGS)

#### Evaluation ######
zcache_dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False)
print(len(zcache_dataloader))
print()

mini_batch_size = 5
num_sets = int(NUM_SAMPLE/mini_batch_size)
true_num_samples = num_sets*mini_batch_size # just take the floor


### For quantitative evaluation
total_err_absRel = 0.0
total_err_squaRel = 0.0
total_err_silog = 0.0
total_err_delta1 = 0.0
total_err_whdr = 0.0
num_evaluated = 0

model.eval()

### Focal length for taskonomy
f = 512.0 ### Hardcoded for taskonomy   


with torch.no_grad():
    for i, data in enumerate(zcache_dataloader):

        batch_size = data['rgb'].shape[0]
        C = data['rgb'].shape[1]
        H = data['rgb'].shape[2]
        W = data['rgb'].shape[3]

        ## Get original image size
        curr_imgh_path = data['A_paths'][0]
        img = cv2.imread(curr_imgh_path)
        img = cv2.resize(img, (448, 448))

        ### Repeat for the number of samples
        num_images = data['rgb'].shape[0]
        data['rgb'] = data['rgb'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
        data['rgb'] = data['rgb'].view(-1, C, H, W)

        ### Iterate over the minibatch
        image_fname = []

        rgb = torch.clone(data['rgb'][0]).permute(1, 2, 0).to("cpu").detach().numpy()
        rgb = 255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb = np.array(rgb, np.int)

        ### Ground truth depth
        # gt_depth = transform_shift_scale(data['gt_depth'].cuda())
        gt_depth = data['gt_depth'].cuda()
        curr_gt = gt_depth[0]
        curr_gt = curr_gt.to("cpu").detach().numpy().squeeze()
        # curr_gt = cv2.resize(curr_gt, (448, 448), interpolation=cv2.INTER_NEAREST)


        if i%50==0 or (i%10==0 and FLAGS.phase_anno != "train") or VISU_ALL:      
            img_name = "image" + str(i)
            cv2.imwrite(os.path.join(temp_fol, img_name+"-raw.png"), rgb)
            image_fname.append(os.path.join(temp_fol, img_name+"-raw.png"))
            img_name = "image" + str(i) + "_gt"
            reconstruct_depth(curr_gt, rgb, gt_fol, img_name, f)

        all_err_absRel = np.zeros((batch_size, mini_batch_size))
        all_err_squaRel = np.zeros((batch_size, mini_batch_size))
        all_err_silog = np.zeros((batch_size, mini_batch_size))
        all_err_delta1 = np.zeros((batch_size, mini_batch_size))
        all_err_whdr = np.zeros((batch_size, mini_batch_size))

        for k in range(num_sets):

            ## Hard coded d_latent
            z = torch.normal(0.0, 1.0, size=(num_images, mini_batch_size, D_LATENT))
            z = z.view(-1, D_LATENT).cuda()

            pred_depth = model.inference(data, z, rescaled=RESCALED)

            for s in range(mini_batch_size):
                curr_pred_depth = pred_depth[s]
                curr_pred_depth = curr_pred_depth.to("cpu").detach().numpy().squeeze() 

                pred_depth_ori = curr_pred_depth

                curr_pred_depth_metric = recover_metric_depth(np.copy(curr_pred_depth), curr_gt)

                # pred_depth_ori = cv2.resize(curr_pred_depth, (H, W))

                img_name = "image" + str(i) + "_" + str(k) + "_" + str(s)
                
                if i%50==0 or (i%10==0 and FLAGS.phase_anno != "train") or VISU_ALL:

                    # save depth
                    plt.imsave(os.path.join(temp_fol, img_name+'-depth.png'), pred_depth_ori, cmap='rainbow') 
                    image_fname.append(os.path.join(temp_fol, img_name+'-depth.png'))

                    ### Output point cloud
                    reconstruct_depth(np.copy(curr_pred_depth), rgb, pc_fol, img_name, f)

                    reconstruct_depth(np.copy(curr_pred_depth_metric), rgb, pc_fol, img_name+"_metric", f)


                    shift, focal_length, depth_scaleinv = reconstruct3D_from_depth(img, pred_depth_ori,
                                                                           shift_model, focal_model)
                    print(focal_length)
                    curr_pred_depth_metric_corrected = recover_metric_depth(depth_scaleinv, curr_gt)

                    reconstruct_depth(depth_scaleinv, rgb, pc_fol, img_name+'-pcd-shiftfocal', focal=focal_length)


                    ### Project into ground truth camera
                    pointcloud = reconstruct_3D(depth_scaleinv, f=focal_length)

                    ### Project to camera given a focal length
                    # print(pointcloud.shape)
                    # print(rgb.shape)
                    pointcloud_visible = get_nonoccluded_points(pointcloud, f, rgb)
                    # print(pointcloud_visible.shape)
                    pointcloud_2d = project_2d(pointcloud_visible, f, rgb)
                    
                    # print(pointcloud)
                    # print()
                    print(pointcloud_2d)
                    print(pointcloud_2d.shape)
                    # print(f)
                    # print(rgb.shape)
                    exit()

                    ### Projected depth map --> with holes
                    width = rgb.shape[0]
                    height = rgb.shape[1]

                    projected_depth = np.zeros((height,width))

                    pointcloud_2d = pointcloud_2d.astype(int)
                    projected_depth[pointcloud_2d[1], pointcloud_2d[0]] = pointcloud_visible[:,2]
                    
                    # print(projected_depth)
                    # print(projected_depth.shape)
                    plt.imsave(os.path.join(temp_fol, img_name+'-depth-projected.png'), projected_depth*1000., cmap='rainbow')

                    ### Grid data interpolate
                    grid_x, grid_y = np.mgrid[0:height, 0:width]
                    interpolated_depth_nearest = griddata(pointcloud_2d.T, pointcloud_visible[:,2], (grid_x, grid_y), method='nearest', fill_value=0.0).T
                    interpolated_depth_linear = griddata(pointcloud_2d.T, pointcloud_visible[:,2], (grid_x, grid_y), method='linear', fill_value=0.0).T
                    interpolated_depth_cubic = griddata(pointcloud_2d.T, pointcloud_visible[:,2], (grid_x, grid_y), method='cubic', fill_value=0.0).T

                    # print(interpolated_depth_linear)
                    # print(interpolated_depth_cubic)
                    # exit()

                    # print(interpolated_depth_nearest.shape)
                    # print(interpolated_depth_linear.shape)
                    # print(interpolated_depth_cubic.shape)

                    plt.imsave(os.path.join(temp_fol, img_name+'-depth-projected-interpolated-nearest.png'), interpolated_depth_nearest, cmap='rainbow')
                    plt.imsave(os.path.join(temp_fol, img_name+'-depth-projected-interpolated-linear.png'), interpolated_depth_linear, cmap='rainbow')
                    plt.imsave(os.path.join(temp_fol, img_name+'-depth-projected-interpolated-cubic.png'), interpolated_depth_cubic, cmap='rainbow')


                    ### Scale with gt depth then output as point cloud
                    curr_interpolated_depth_metric_nearest = recover_metric_depth(interpolated_depth_nearest, curr_gt)
                    curr_interpolated_depth_metric_linear = recover_metric_depth(interpolated_depth_linear, curr_gt)
                    curr_interpolated_depth_metric_cubic = recover_metric_depth(interpolated_depth_cubic, curr_gt)

                    reconstruct_depth(curr_interpolated_depth_metric_nearest, rgb, pc_fol, img_name+"_metric_nearest", f)
                    reconstruct_depth(curr_interpolated_depth_metric_linear, rgb, pc_fol, img_name+"_metric_linear", f)
                    reconstruct_depth(curr_interpolated_depth_metric_cubic, rgb, pc_fol, img_name+"_metric_cubic", f)

                    exit()

                ### Align prediction and compute qualitative results
                # curr_pred_depth_metric = recover_metric_depth(curr_pred_depth, curr_gt)
                err_absRel, err_squaRel, err_silog, err_delta1, err_whdr = evaluate_rel_err(curr_pred_depth_metric, curr_gt)

                all_err_absRel[:, k*mini_batch_size + s] = err_absRel
                all_err_squaRel[:, k*mini_batch_size + s] = err_squaRel
                all_err_silog[:, k*mini_batch_size + s] = err_silog
                all_err_delta1[:, k*mini_batch_size + s] = err_delta1
                all_err_whdr[:, k*mini_batch_size + s] = err_whdr
            #######

        ### Quantitative Results
        idx_to_take = np.argmin(all_err_absRel, axis=-1)[0]

        if all_err_absRel[0][idx_to_take] > 0:
            total_err_absRel += all_err_absRel[0][idx_to_take]
            total_err_squaRel += all_err_squaRel[0][idx_to_take]
            total_err_silog += all_err_silog[0][idx_to_take]
            total_err_delta1 += all_err_delta1[0][idx_to_take]
            total_err_whdr += all_err_whdr[0][idx_to_take]
            num_evaluated += 1

        #######################

        if i%50==0 or (i%10==0 and FLAGS.phase_anno != "train") or VISU_ALL:
            ### Collate and output to a single image
            height = H
            width = W    

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
        # print(output_image_filename)

        if i%100==0:
            print("Finished "+str(i)+"/"+str(len(zcache_dataloader))+".")

mean_err_absRel = total_err_absRel/float(num_evaluated)
mean_err_squaRel = total_err_squaRel/float(num_evaluated)
mean_err_silog = total_err_silog/float(num_evaluated)
mean_err_delta1 = total_err_delta1/float(num_evaluated)
mean_err_whdr = total_err_whdr/float(num_evaluated)

log_string("=" * 20)
log_string("")
log_string("Num evaluated= "+str(num_evaluated))
log_string("")
log_string("Mean Err AbsRel= "+str(mean_err_absRel))
log_string("")             
log_string("Mean Err SquareRel = "+str(mean_err_squaRel))
log_string("")     
log_string("Mean Err SiLog= "+str(mean_err_silog))
log_string("")                
log_string("Mean Scale Err = "+str(mean_err_delta1))
log_string("") 
log_string("Mean WHDR = "+str(mean_err_whdr))
log_string("")           

print("Done.")




