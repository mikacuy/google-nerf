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
        batch_size=4,
        num_workers=0,
        shuffle=False)

    print(len(zcache_dataloader))
    print()

    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    tmp_i = 0
    try:
        ### Set network to eval mode
        model.eval()


        with torch.no_grad():
            for i, data in enumerate(zcache_dataloader):     
                batch_size = data['rgb'].shape[0]
                C = data['rgb'].shape[1]
                H = data['rgb'].shape[2]
                W = data['rgb'].shape[3]

                pred_depth = model.inference(data)

                for s in range(batch_size):
                    curr_pred_depth = pred_depth[s]

                    curr_pred_depth = curr_pred_depth.to("cpu").detach().numpy().squeeze() 

                    pred_depth_ori = cv2.resize(curr_pred_depth, (H, W))

                    # if GT depth is available, uncomment the following part to recover the metric depth
                    #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

                    img_name = "image" + str(i) + "_" + str(s)
                    rgb = data['rgb'][s].permute(1, 2, 0).to("cpu").detach().numpy()

                    rgb = 255 * (rgb - rgb.min()) / (rgb.max() - rgb.min())
                    rgb = np.array(rgb, np.int)

                    cv2.imwrite(os.path.join(visu_dir, img_name+"-raw.png"), rgb)
                    # save depth
                    plt.imsave(os.path.join(visu_dir, img_name+'-depth.png'), pred_depth_ori, cmap='rainbow')
                    cv2.imwrite(os.path.join(visu_dir, img_name+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))


                if i%10==0:
                    print("Finished "+str(i)+"/"+str(len(zcache_dataloader))+".")

                    torch.cuda.empty_cache()



    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)
    finally:
        if train_args.use_tfboard and main_process(dist=train_args.distributed, rank=train_args.global_rank):
            tblogger.close()


def main_worker(local_rank: int, ngpus_per_node: int, train_args, val_args):
    train_args.global_rank = train_args.node_rank * ngpus_per_node + local_rank
    train_args.local_rank = local_rank
    val_args.global_rank = train_args.global_rank
    val_args.local_rank = local_rank
    merge_cfg_from_file(train_args)

    global logger

    ### Override and specify log_dir name
    cfg.TRAIN.RUN_NAME = train_args.run_name
    cfg.TRAIN.OUTPUT_DIR = './outputs'
    # Dir for checkpoint and logs
    cfg.TRAIN.LOG_DIR = os.path.join(cfg.TRAIN.OUTPUT_DIR, cfg.TRAIN.RUN_NAME)
    log_output_dir = cfg.TRAIN.LOG_DIR
    visu_dir = os.path.join(log_output_dir, "visu_singlehyp")

    if visu_dir:
        try:
            os.makedirs(visu_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    tblogger = None

    logger = setup_distributed_logger("lib", log_output_dir, local_rank, cfg.TRAIN.RUN_NAME[:-1]+ '_eval.txt')


    # # init
    # if train_args.distributed:
    #     torch.cuda.set_device(local_rank)
    #     dist.init_process_group(backend='nccl',
    #                             init_method=train_args.dist_url,
    #                             world_size=train_args.world_size,
    #                             rank=train_args.global_rank)


    # load model
    model = RelDepthModel()


    # if train_args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model.cuda(), device_ids=[local_rank], output_device=local_rank)
    # else:
    #     model = torch.nn.DataParallel(model.cuda())

    model.cuda()
    val_err = [{'abs_rel': 0, 'whdr': 0}]

    # Print configs and logs
    print_options(train_args, logger)
    print_options(val_args, logger)

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

    print("To start training...")

    do_train(train_dataset,
             val_dataset,
             train_args,
             model,
             save_to_disk,
             None,
             None,
             val_err,
             logger,
             tblogger,
             visu_dir)

def main():
    # Train args
    train_opt = TrainOptions()
    train_args = train_opt.parse()

    # Validation args
    val_opt = ValOptions()
    val_args = val_opt.parse()
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
