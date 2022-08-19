import math
import traceback
import errno
import os, sys
import torch.distributed as dist
import torch.multiprocessing as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))

from multiprocessing.sharedctypes import Value
from data.load_dataset_distributed import MultipleDataLoaderDistributed, MultipleDatasetDistributed
from lib.models.multi_depth_model_auxiv2 import *
from lib.configs.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_rel_depth_err, recover_metric_depth
from lib.utils.lr_scheduler_custom import make_lr_scheduler
from lib.utils.comm import is_pytorch_1_1_0_or_later, get_world_size
from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_distributed_logger, SmoothedValue
from tools.parse_arg_base import print_options
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions

## for dataloaders
import torch.utils.data
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.pyplot as plt

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

def main_process(dist, rank) -> bool:
    return not dist or (dist and rank == 0)

def increase_sample_ratio_steps(step, base_ratio=0.1, step_size=10000):
    ratio = min(base_ratio * (int(step / step_size) + 1), 1.0)
    return ratio

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def val(val_dataloader, model):
    """
    Validate the model.
    """
    print('validating...')
    smoothed_absRel = SmoothedValue(len(val_dataloader))
    smoothed_whdr = SmoothedValue(len(val_dataloader))
    smoothed_criteria = {'err_absRel': smoothed_absRel, 'err_whdr': smoothed_whdr}
    for i, data in enumerate(val_dataloader):
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['pred_depth'])

        pred_depth_resize = cv2.resize(pred_depth.cpu().numpy(), (torch.squeeze(data['gt_depth']).shape[1], torch.squeeze(data['gt_depth']).shape[0]))
        pred_depth_metric = recover_metric_depth(pred_depth_resize, data['gt_depth'])
        smoothed_criteria = validate_rel_depth_err(pred_depth_metric, data['gt_depth'], smoothed_criteria, scale=1.0)
    return {'abs_rel': smoothed_criteria['err_absRel'].GetGlobalAverageValue(),
            'whdr': smoothed_criteria['err_whdr'].GetGlobalAverageValue()}


def do_train(train_dataset, val_dataset, train_args,
             model, save_to_disk,
             scheduler, optimizer, val_err,
             logger, tblogger=None, visu_dir=None):
    # training status for logging
    print(visu_dir)


    ### Dataloader unshuffled to cache z-codes
    zcache_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False)

    print(len(zcache_dataloader))
    print()

    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    tmp_i = 0

    ### Set network to eval mode
    model.eval()


    with torch.no_grad():

        # all_losses = []
        # for i, data in enumerate(zcache_dataloader):     
        #     batch_size = data['rgb'].shape[0]
        #     C = data['rgb'].shape[1]
        #     H = data['rgb'].shape[2]
        #     W = data['rgb'].shape[3]

        #     pred_depth, losses = model.inference(data, return_loss=True)
        #     loss_dict, total_loss = losses
        #     total_loss = total_loss.to("cpu").detach().numpy()

        #     per_pixel_ilnr = loss_dict["ilnr_per_pixel"].to("cpu").detach().numpy()


        #     for s in range(batch_size):
        #         all_losses.append(total_loss[s])

        #     if i%10==0:
        #         print("Finished "+str(i)+"/"+str(len(zcache_dataloader))+".")

        #     torch.cuda.empty_cache()


        # all_losses = np.array(all_losses)
        # sorted_idx = np.argsort(-all_losses) ## sort and get the biggest loss
        # sorted_losses = np.sort(all_losses)

        # fname = "val_sortedlosses_indices.npy"
        # np.save(os.path.join(visu_dir, "..", fname), sorted_idx)

        # fname = "val_alllosses.npy"
        # np.save(os.path.join(visu_dir, "..", fname), all_losses)

        # ###Output graph of the losses
        # plt.plot(np.arange(len(all_losses)), sorted_losses)
        # plt.ylabel('Sorted Total loss')
        # plt.savefig(os.path.join(visu_dir, "..", "val_sorted_losses.png"))

        fname_sorted_idx = "val_sortedlosses_indices.npy"
        sorted_idx = np.load(os.path.join(visu_dir, "..", fname_sorted_idx))

        for i in range(len(sorted_idx)):     
            data = val_dataset[sorted_idx[i]]
            # data = val_dataset[sorted_idx[-i-1]]

            ### Expand because dataloader was removed
            for key in data.keys():
                if (torch.is_tensor(data[key])):
                    data[key] = data[key].unsqueeze(0).cuda()
                if key == "quality_flg":
                    data[key] = torch.from_numpy(data[key]).unsqueeze(0)


            batch_size = data['rgb'].shape[0]
            C = data['rgb'].shape[1]
            H = data['rgb'].shape[2]
            W = data['rgb'].shape[3]

            pred_depth, losses = model.inference(data, return_loss=True)
            loss_dict, total_loss = losses
            total_loss = total_loss.to("cpu").detach().numpy()

            print(loss_dict)
            per_pixel_ilnr = loss_dict["ilnr_per_pixel"].to("cpu").detach().numpy()


            for s in range(batch_size):
                ## GT
                gt_depth = transform_shift_scale(data['gt_depth'].cuda())
                gt_depth = gt_depth.to("cpu").detach().numpy().squeeze()
                print(gt_depth.shape)
                gt_depth = cv2.resize(gt_depth, (448, 448))
                print(gt_depth)
                exit()

                curr_pred_depth = pred_depth[s]

                curr_pred_depth_metric = recover_metric_depth(curr_pred_depth, gt_depth)
                curr_pred_depth = curr_pred_depth.to("cpu").detach().numpy().squeeze() 

                pred_depth_ori = cv2.resize(curr_pred_depth, (H, W))
                curr_pred_depth_metric = cv2.resize(curr_pred_depth_metric, (H, W))

                # if GT depth is available, uncomment the following part to recover the metric depth
                #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

                img_name = "image" + str(i) + "_" + str(s)

                ### Reading function
                rgb = cv2.imread(data['A_paths'])
                rgb = cv2.resize(rgb, (448, 448))

                cv2.imwrite(os.path.join(visu_dir, img_name+"-a_inputimage.png"), rgb)
                # save depth
                plt.imsave(os.path.join(visu_dir, img_name+'e-depthscaled.png'), pred_depth_ori, cmap='rainbow')
                cv2.imwrite(os.path.join(visu_dir, img_name+'c-preddepth_raw.png'), (curr_pred_depth_metric/10. * 60000).astype(np.uint16))


                cv2.imwrite(os.path.join(visu_dir, img_name+'b-gtdepth_raw.png'), (gt_depth/10. * 60000).astype(np.uint16))
                cv2.imwrite(os.path.join(visu_dir, img_name+'d-predraw_nonmetric.png'), (pred_depth_ori/10. * 60000.).astype(np.uint16))

                ### Per Pixel ILNR vis
                curr_ilnr = per_pixel_ilnr[s].squeeze()
                curr_ilnr = cv2.resize(curr_ilnr, (H, W))

                # cv2.imwrite(os.path.join(visu_dir, img_name+'-ilnr-raw.png'), (-curr_ilnr/5.0 * 60000).astype(np.uint16))
                
                plt.imsave(os.path.join(visu_dir, img_name+'f-ilnr.png'), curr_ilnr, cmap='rainbow', vmin=0, vmax=10)


            if i%10==0:
                print("Finished "+str(i)+"/"+str(len(zcache_dataloader))+".")

            if i>20:
                break



def main_worker(local_rank: int, ngpus_per_node: int, train_args, val_args):
    train_args.global_rank = train_args.node_rank * ngpus_per_node + local_rank
    train_args.local_rank = local_rank
    val_args.global_rank = train_args.global_rank
    val_args.local_rank = local_rank
    merge_cfg_from_file(train_args)

    ### Override and specify log_dir name
    cfg.TRAIN.RUN_NAME = train_args.run_name
    log_output_dir = cfg.TRAIN.RUN_NAME

    if log_output_dir:
        try:
            os.makedirs(log_output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    visu_dir = os.path.join(log_output_dir, "visu_val")

    if visu_dir:
        try:
            os.makedirs(visu_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # load model
    model = RelDepthModel()


    model.cuda()
    val_err = [{'abs_rel': 0, 'whdr': 0}]

    print(train_args.phase)
    print(val_args.phase)
    # exit()

    train_dataset = MultipleDatasetDistributed(train_args)
    val_dataset = MultipleDatasetDistributed(val_args)

    print("Datasets:")
    print("Train")
    print(train_args.dataset_list)
    print(len(train_dataset))
    print()
    print("Val:")    
    print(val_args.dataset_list)
    print(len(val_dataset))
    # print(val_sample_size)
    print("====================")

    ### Load model from run_name
    if train_args.load_ckpt:
        if os.path.isfile(train_args.load_ckpt):
            print("loading checkpoint %s" % train_args.load_ckpt)
            checkpoint = torch.load(train_args.load_ckpt)
            model_dict = model.state_dict()

            ### Check is it is LeReS pretrained model, loading is different

            if "Minist_Test/res101.pth" in train_args.load_ckpt:
                ### Loading pretrained model
                # Filter out unnecessary keys
                # print(model_dict.keys())
                # print()
                # print(checkpoint['depth_model'].keys())
                # print()

                checkpoint['depth_model'] = strip_prefix_if_present(checkpoint['depth_model'], "module.")

                depth_keys = {k: v for k, v in checkpoint['depth_model'].items() if k in model_dict} ## <--- some missing keys in the loaded model from the given model
                print(len(depth_keys))

                # Overwrite entries in the existing state dict
                model_dict.update(depth_keys) 
                print("Loaded from pretrained LeReS.")

            # Load the new state dict
            model.load_state_dict(model_dict)

            del checkpoint
            torch.cuda.empty_cache()

            print("Model loaded.")

    print_configs(cfg)

    save_to_disk = main_process(train_args.distributed, local_rank)

    do_train(train_dataset,
             val_dataset,
             train_args,
             model,
             save_to_disk,
             None,
             None,
             val_err,
             None,
             None,
             visu_dir)

def main():
    # Train args
    train_opt = TrainOptions()
    train_args = train_opt.parse()

    # Validation args
    val_opt = ValOptions()
    val_args = val_opt.parse()

    ## Override for dataloader
    val_args.phase_anno = "val"
    val_args.phase = "test"


    val_args.batchsize = 1
    val_args.thread = 0

    if 'Holopix50k' in val_args.dataset_list:
        val_args.dataset_list.remove('Holopix50k')

    print('Using PyTorch version: ', torch.__version__, torch.version.cuda)
    ngpus_per_node = torch.cuda.device_count()
    train_args.world_size = ngpus_per_node * train_args.nnodes
    val_args.world_size = ngpus_per_node * train_args.nnodes
    train_args.distributed = ngpus_per_node > 1

    # Randomize args.dist_url to avoid conflicts on same machine
    train_args.dist_url = train_args.dist_url + str(os.getpid() % 100).zfill(2)

    if train_args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, train_args, val_args))
    else:
        main_worker(0, ngpus_per_node, train_args, val_args)


if __name__=='__main__':
    main()
