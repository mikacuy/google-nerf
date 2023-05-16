'''
Mikaela Uy
Usable eval script as of Aug 23, 2022
Evaluates on images sorted by LeReS losses
'''
import math
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
from data.load_dataset_distributed import MultipleDataLoaderDistributed, MultipleDatasetDistributed
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

import argparse
from PIL import Image

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="log_0726_lrfixed_001/", help="path to the log directory", type=str)
parser.add_argument("--ckpt", default="epoch80_step30375.pth", help="checkpoint", type=str)

parser.add_argument('--dump_dir', default= "dump_lerescimle_0726_lrfixed_001_newvisu/", type=str)

### For the dataset
parser.add_argument('--phase', type=str, default='test', help='Training flag')
parser.add_argument('--phase_anno', type=str, default='train', help='Annotations file name')
parser.add_argument('--dataset_list', default=["taskonomy"], nargs='+', help='The names of multiple datasets')
parser.add_argument('--dataset', default='multi', help='Dataset loader name')
parser.add_argument('--dataroot', default='/orion/downloads/coordinate_mvs/', help='Root dir for dataset')
parser.add_argument('--loss_mode', type=str, default='_ranking-edge_pairwise-normal-regress-edge_msgil-normal_meanstd-tanh_pairwise-normal-regress-plane_', help='losses to use')

parser.add_argument('--backbone', default= "resnext101", type=str)
parser.add_argument('--d_latent', default= 32, type=int)
parser.add_argument('--num_samples', default= 5, type=int)
parser.add_argument('--rescaled', default=True, type=bool)

parser.add_argument('--ada_version', default= "v2", type=str)
parser.add_argument('--cimle_version', default= "enc", type=str)


FLAGS = parser.parse_args()
LOG_DIR = FLAGS.logdir
CKPT = FLAGS.ckpt

ADA_VERSION = FLAGS.ada_version
CIMLE_VERSION = FLAGS.cimle_version


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

print(CIMLE_VERSION)
print(ADA_VERSION)
print("===================")
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
print(len(dataset))
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

### Load numpy array of sorted indices
vanilla_leres_dumpdir = "dump_leres_vanilla_losssorted/"
fname_sorted_idx = FLAGS.phase_anno + "_sortedlosses_indices.npy"
sorted_idx = np.load(os.path.join(vanilla_leres_dumpdir, fname_sorted_idx))

fname_all_losses = FLAGS.phase_anno + "_alllosses.npy"
all_losses = np.load(os.path.join(vanilla_leres_dumpdir, fname_all_losses))

with torch.no_grad():
    for i in range(len(sorted_idx)):

        if i>20:
            break

        # data = dataset[sorted_idx[i]]    ## worst samples
        data = dataset[sorted_idx[-i-1]]    ## best samples

        ### Expand because dataloader was removed
        for key in data.keys():
            if (torch.is_tensor(data[key])):
                data[key] = data[key].unsqueeze(0)
            if key == "quality_flg":
                data[key] = torch.from_numpy(data[key]).unsqueeze(0)


        batch_size = data['rgb'].shape[0]
        C = data['rgb'].shape[1]
        H = data['rgb'].shape[2]
        W = data['rgb'].shape[3]

        ### Repeat for the number of samples
        num_images = data['rgb'].shape[0]
        data['rgb'] = data['rgb'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
        data['rgb'] = data['rgb'].view(-1, C, H, W)
        data['depth'] = data['depth'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
        data['depth'] = data['depth'].view(-1, 1, H, W)            
        data['quality_flg'] = data['quality_flg'].unsqueeze(1).repeat(1,mini_batch_size)
        data['quality_flg'] = data['quality_flg'].view(-1)
        data['focal_length'] = data['focal_length'].unsqueeze(1).repeat(1,mini_batch_size)
        data['focal_length'] = data['focal_length'].view(-1)
        data['planes'] = data['planes'].unsqueeze(1).repeat(1,mini_batch_size,1,1)
        data['planes'] = data['planes'].view(-1, H, W)


        ### Iterate over the minibatch
        image_fname = []

        rgb = cv2.imread(data['A_paths'])
        rgb = cv2.resize(rgb, (448, 448))
     
        img_name = "image" + str(i)
        raw_img_name = os.path.join(temp_fol, img_name+"-raw.png")
        cv2.imwrite(raw_img_name, rgb)

        ### Hardcoded focal length for taskonomy
        f = 512.0 ### Hardcoded for taskonomy

        gt_depth = data['depth'][0]
        # print(gt_depth)
        # print(gt_depth.shape)
        # print()

        curr_gt = transform_shift_scale(gt_depth.cuda())
        curr_gt = curr_gt.to("cpu").detach().numpy().squeeze()
        curr_gt = cv2.resize(curr_gt, (448, 448))

        # print(curr_gt)
        # print(curr_gt.shape)
        # exit()

        depth_name = os.path.join(temp_fol, img_name+'gtdepth.png')
        cv2.imwrite(depth_name, (curr_gt/10. * 60000).astype(np.uint16))

        img_name = "image" + str(i) + "_gtdepth"
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

            ##########
            pred_depth, losses = model.inference(data, z, rescaled=RESCALED, return_loss=True)
            loss_dict, total_loss = losses
            total_loss = total_loss.to("cpu").detach().numpy()

            per_pixel_ilnr = loss_dict["ilnr_per_pixel"].to("cpu").detach().numpy()
            ##########

            all_depthmetric_names = []
            all_depthscaled_names = []
            all_loss_names = []

            for s in range(mini_batch_size):
                curr_pred_depth = pred_depth[s]

                curr_pred_depth_metric = recover_metric_depth(curr_pred_depth, curr_gt) ### scaled and shifted by least squares fitting
                curr_pred_depth = curr_pred_depth.to("cpu").detach().numpy().squeeze() 

                pred_depth_ori = cv2.resize(curr_pred_depth, (H, W))
                curr_pred_depth_metric = cv2.resize(curr_pred_depth_metric, (H, W))

                name_depthscaled = "image" + str(i) + "_" + str(k) + "_" + str(s) + "depthscaled.png"
                plt.imsave(os.path.join(temp_fol, name_depthscaled), pred_depth_ori, cmap='rainbow')

                name_depthmetric = "image" + str(i) + "_" + str(k) + "_" + str(s) + "depthmetric.png"
                cv2.imwrite(os.path.join(temp_fol, name_depthmetric), (curr_pred_depth_metric/10. * 60000).astype(np.uint16))

                ### Per Pixel ILNR vis
                curr_ilnr = per_pixel_ilnr[s].squeeze()
                curr_ilnr = cv2.resize(curr_ilnr, (H, W)) 

                name_depthloss = "image" + str(i) + "_" + str(k) + "_" + str(s) + "depthloss.png"                               
                plt.imsave(os.path.join(temp_fol, name_depthloss), curr_ilnr, cmap='rainbow', vmin=0, vmax=10)
 

                img_name = "image_pred" + str(sorted_idx[i]) + "_" + str(k) + "_" + str(s)
                image_fname.append(os.path.join(temp_fol, img_name+'-depth.png'))

                ### Output point cloud
                f = 512.0 ### Hardcoded for taskonomy
                reconstruct_depth(curr_pred_depth_metric, rgb, pc_fol, img_name, f)


                all_depthmetric_names.append(name_depthmetric)
                all_depthscaled_names.append(name_depthscaled)
                all_loss_names.append(name_depthloss)


        #     #### Fix this !!!
        #     err_absRel, err_squaRel, err_silog, err_delta1, err_whdr = evaluate_rel_err(curr_pred_depth_scaled, curr_gt)
        #     if err_absRel <0 :
        #         ## Error skip
        #         all_err_absRel[:, k*mini_batch_size + s] = 1000000.
        #     else:
        #         all_err_absRel[:, k*mini_batch_size + s] = err_absRel
        #         all_err_squaRel[:, k*mini_batch_size + s] = err_squaRel
        #         all_err_silog[:, k*mini_batch_size + s] = err_silog
        #         all_err_delta1[:, k*mini_batch_size + s] = err_delta1
        #         all_err_whdr[:, k*mini_batch_size + s] = err_whdr
        #     #######

        # ### Quantitative Results
        # # print(all_err_absRel)
        # idx_to_take = np.argmin(all_err_absRel, axis=-1)[0]
        # # print(idx_to_take)

        # if all_err_absRel[0][idx_to_take] > 0:
        #     total_err_absRel += all_err_absRel[0][idx_to_take]
        #     total_err_squaRel += all_err_squaRel[0][idx_to_take]
        #     total_err_silog += all_err_silog[0][idx_to_take]
        #     total_err_delta1 += all_err_delta1[0][idx_to_take]
        #     total_err_whdr += all_err_whdr[0][idx_to_take]
        #     num_evaluated += 1

        #######################

        ### Collate and output to a single image
        height = H
        width = W    

        #Output to a single image
        new_im = Image.new('RGB', (width*(1+NUM_SAMPLE), height))

        image_fname = []
        image_fname.append(raw_img_name)
        for fname in all_depthscaled_names:
            image_fname.append(os.path.join(temp_fol,fname))

        images = []
        for fname in image_fname:
            images.append(Image.open(fname))        

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += width

        output_image_filename = os.path.join(DUMP_DIR, str(i) +'_deptha_scaled.png')
        new_im.save(output_image_filename) 

        ######################

        new_im = Image.new('L', (width*(1+NUM_SAMPLE), height))
        image_fname = []
        image_fname.append(depth_name)
        for fname in all_depthmetric_names:
            image_fname.append(os.path.join(temp_fol,fname))

        images = []
        for fname in image_fname:
            np_image = np.array(Image.open(fname)).astype("uint16")
            cvuint8 = cv2.convertScaleAbs(np_image, alpha=(255.0/65535.0))
            images.append(Image.fromarray(cvuint8))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += width

        output_image_filename = os.path.join(DUMP_DIR, str(i) +'_depthb_alignedgt.png')
        new_im.save(output_image_filename) 

        ######################

        new_im = Image.new('RGBA', (width*(NUM_SAMPLE), height))
        image_fname = []
        for fname in all_loss_names:
            image_fname.append(os.path.join(temp_fol,fname))

        images = []
        for fname in image_fname:
            images.append(Image.open(fname))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += width


        output_image_filename = os.path.join(DUMP_DIR, str(i) +'_ilnrerror.png')
        new_im.save(output_image_filename) 

        if i%100==0:
            print("Finished "+str(i)+"/"+str(len(sorted_idx))+".")



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























