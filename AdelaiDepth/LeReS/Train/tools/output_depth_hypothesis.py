'''
Mikaela Uy
Usable eval script as of Aug 25, 2022
NerF evaluation script
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


parser = argparse.ArgumentParser()
# parser.add_argument("--logdir", default="log_0726_lrfixed_001/", help="path to the log directory", type=str)
# parser.add_argument("--ckpt", default="epoch104_step39375.pth", help="checkpoint", type=str)

parser.add_argument("--logdir", default="log_0825_encv2_noaug_noshuffle_s12/", help="path to the log directory", type=str)
parser.add_argument("--ckpt", default="epoch56_step0.pth", help="checkpoint", type=str)

# parser.add_argument("--logdir", default="log_finetune_scannet0653_0825/", help="path to the log directory", type=str)
# parser.add_argument("--ckpt", default="epoch9_step0.pth", help="checkpoint", type=str)

parser.add_argument('--dump_dir', default= "dump_0826_pretrained_dd_scene0710_train/", type=str)
# parser.add_argument('--dump_dir', default= "dump_0826_pretrained_dd_scene0758_train/", type=str)
# parser.add_argument('--dump_dir', default= "dump_0826_pretrained_dd_scene0781_train/", type=str)

### For the dataset
parser.add_argument('--phase', type=str, default='test', help='Training flag')

# ### Scannet sample scene
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/mika_processed/scene0653_00/', help='Root dir for dataset')
parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/scenes/scene0710_00/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/scenes/scene0758_00/train/', help='Root dir for dataset')
# parser.add_argument('--dataroot', default='/orion/group/scannet_v2/dense_depth_priors/scenes/scene0781_00/train/', help='Root dir for dataset')

### Nerf
# parser.add_argument('--dataroot', default='/orion/group/NSVF/Synthetic_NeRF/Lego', help='Root dir for dataset')

parser.add_argument('--backbone', default= "resnext101", type=str)
parser.add_argument('--d_latent', default= 32, type=int)
parser.add_argument('--num_samples', default= 20, type=int)
parser.add_argument('--rescaled', default=False, type=bool)

parser.add_argument('--ada_version', default= "v2", type=str)
parser.add_argument('--cimle_version', default= "enc", type=str)
parser.add_argument('--import_from_logdir', default=False, type=bool)
parser.add_argument('--visu_all', default=False, type=bool)
parser.add_argument('--seed_num', default= 0, type=int)

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

    # pred_metric[~mask] = 0.

    return pred_metric

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


### Dataset

################################
### Old not using the dataloader
################################
# if not IS_NSVF:
#     image_dir = os.path.join(FLAGS.dataroot, FLAGS.scenename, "rgb")
#     depth_dir = os.path.join(FLAGS.dataroot, FLAGS.scenename, "depth")
# else:
#     image_dir = os.path.join(FLAGS.dataroot, FLAGS.scenename, "leres_cimle_v1", "rgb")
#     depth_dir = os.path.join(FLAGS.dataroot, FLAGS.scenename, "leres_cimle_v1", "depth")


# imgs_list = os.listdir(image_dir)
# imgs_list.sort()

# depth_list = os.listdir(depth_dir)
# depth_list.sort()

# imgs_path = [os.path.join(image_dir, i) for i in imgs_list]
# print(len(imgs_path))

# depth_path = [os.path.join(depth_dir, i) for i in depth_list]
# print(len(imgs_path))

# if len(imgs_path) !=  len(depth_path):
#     print("ERROR. Number of images and depth maps don't match.")
#     exit()
################################

### Dataloader
# datapath = os.path.join(FLAGS.dataroot, FLAGS.scenename)
datapath = FLAGS.dataroot

if IS_NSVF:
    dataset_name = "nsvf"
else:
    dataset_name = "scannet"
dataset = FinetuneDataset(datapath, dataset_name, is_nsvf=IS_NSVF, split="test", data_aug=False)


### Create output dir for the multiple hypothesis
hypothesis_outdir = os.path.join(FLAGS.dataroot, "leres_cimle", DUMP_DIR)
if not os.path.exists(hypothesis_outdir): os.makedirs(hypothesis_outdir)

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
total_err_absRel = 0.0
total_err_squaRel = 0.0
total_err_silog = 0.0
total_err_delta1 = 0.0
total_err_whdr = 0.0
num_evaluated = 0

model.eval()

### Focal length for scannet
if not IS_NSVF:
    f = 577.870605
else:
    ### Check this for other models
    f = 1111.111

# with torch.no_grad():
#     for i in range(len(imgs_path)):
        # curr_img_path = imgs_path[i]
        # curr_depth_path = depth_path[i]
        
        # ## Load iamge
        # rgb = cv2.imread(curr_img_path)
        # rgb_c = rgb.copy()
        # gt_depth = None
        # A_resize = cv2.resize(rgb_c, (448, 448), interpolation=cv2.INTER_LINEAR)
        # img_torch = scale_torch(A_resize)[None, :, :, :]

        # ## Load depth
        # depth_img = np.array(imageio.imread(curr_depth_path))

        # if not IS_NSVF:
        #     ## Scannet depth
        #     depth_img = depth_img.astype(float)/1000.
        #     depth_img = (depth_img / depth_img.max() * 60000).astype(np.uint16)
        # else:
        #     depth_img = remap_color_to_depth(depth_img)
        #     depth_img = depth_img.astype(float)
        #     depth_img = (depth_img / depth_img.max() * 60000).astype(np.uint16)

        # data['gt_depth'] = depth_img
        # curr_gt = cv2.resize(depth_img, (448, 448), interpolation=cv2.INTER_NEAREST)
        # gt_mask = curr_gt < 1e-8


        # data = {}
        # data['rgb'] = img_torch

with torch.no_grad():
    for i, data in enumerate(zcache_dataloader):

        batch_size = data['rgb'].shape[0]
        C = data['rgb'].shape[1]
        H = data['rgb'].shape[2]
        W = data['rgb'].shape[3]

        ### Repeat for the number of samples
        num_images = data['rgb'].shape[0]
        data['rgb'] = data['rgb'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
        data['rgb'] = data['rgb'].view(-1, C, H, W)

        gt_depth = data['gt_depth'].cuda()

        if len(gt_depth.shape) == 3:
            curr_gt = gt_depth[0]
        else:
            curr_gt = gt_depth

        curr_gt = curr_gt.to("cpu").detach().numpy().squeeze()
        curr_gt = cv2.resize(curr_gt, (448, 448), interpolation=cv2.INTER_NEAREST)

        rgb = torch.clone(data['rgb'][0]).permute(1, 2, 0).to("cpu").detach().numpy() 
        rgb = rgb[:, :, ::-1] ## dataloader is bgr
        rgb = 255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())
        rgb = np.array(rgb, np.int)


        ### Raw gt depth ###
        curr_depth_path = data['B_paths'][0]
        depth_img = np.array(imageio.imread(curr_depth_path))

        if not IS_NSVF:
            ## Scannet depth
            depth_img = depth_img.astype(float)/1000.
        else:
            depth_img = remap_color_to_depth(depth_img)
            depth_img = depth_img.astype(float)

        orig_shape = depth_img.shape
        depth_img = cv2.resize(depth_img, (448, 448), interpolation=cv2.INTER_NEAREST)
        # print(depth_img)

        ### Iterate over the minibatch
        image_fname = []

        if i%10==0  or VISU_ALL:      
            img_name = "image" + str(i)
            cv2.imwrite(os.path.join(temp_fol, img_name+"-raw.png"), rgb)
            image_fname.append(os.path.join(temp_fol, img_name+"-raw.png"))
            img_name = "image" + str(i) + "_gt"
            reconstruct_depth(curr_gt, rgb, gt_fol, img_name, f)

        all_err_absRel = np.zeros((batch_size, mini_batch_size*num_sets))
        all_err_squaRel = np.zeros((batch_size, mini_batch_size*num_sets))
        all_err_silog = np.zeros((batch_size, mini_batch_size*num_sets))
        all_err_delta1 = np.zeros((batch_size, mini_batch_size*num_sets))
        all_err_whdr = np.zeros((batch_size, mini_batch_size*num_sets))

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
                
                if i%10==0  or VISU_ALL:
                    # save depth
                    plt.imsave(os.path.join(temp_fol, img_name+'-depth.png'), pred_depth_ori, cmap='rainbow')
                    image_fname.append(os.path.join(temp_fol, img_name+'-depth.png'))

                    ### Output point cloud
                    reconstruct_depth(curr_pred_depth, rgb, pc_fol, img_name, f)

                ### Align prediction and compute qualitative results
                curr_pred_depth_metric = recover_metric_depth(curr_pred_depth, curr_gt)

                ## Raw depth
                curr_pred_depth_raw = recover_metric_depth(curr_pred_depth, depth_img)
                curr_pred_depth_raw = cv2.resize(curr_pred_depth_raw, orig_shape) ### check this resize function 

                # print(curr_pred_depth_raw)
                # print(curr_pred_depth_raw.shape)
                # print()

                ### Save output hypothesis ###
                curr_rbg_name = data['A_paths'][0]
                # print(curr_rbg_name)
                fname = curr_rbg_name.split("/")[-1][:-4] + "_" + str(k*mini_batch_size+s) + ".npy"

                outfname = os.path.join(hypothesis_outdir, fname)
                np.save(outfname, np.array(curr_pred_depth_raw))

                # ## Check output --> debug
                # depth = np.load(outfname).astype(np.float64)
                # print(depth)
                # print(depth.shape)
                # print(outfname)
                ##############################

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








