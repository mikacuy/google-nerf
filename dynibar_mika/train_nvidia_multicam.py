'''
CUDA_VISIBLE_DEVICES=2,3 python train_nvidia.py --config configs_nvidia/train_playground_long.txt
python train_nvidia.py --config configs_nvidia/train_umbrella_long.txt

CUDA_VISIBLE_DEVICES=2,3 python train_nvidia.py --config configs_nvidia/train_jumping_long.txt
CUDA_VISIBLE_DEVICES=0,1 python train_nvidia.py --config configs_nvidia/train_skating_long.txt

python train_nvidia.py --config configs_nvidia/train_balloon2_long.txt
CUDA_VISIBLE_DEVICES=2,3 python train_nvidia.py --config configs_nvidia/train_balloon2_long.txt

CUDA_VISIBLE_DEVICES=0,1 python train_nvidia.py --config configs_nvidia/train_truck_long.txt
python train_nvidia.py --config configs_nvidia/train_dynamicFace_long.txt

python train_nvidia.py --config configs_nvidia/train_jumping_long.txt

'''

import os
import time
import numpy as np
import shutil
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image
from ibrnet.model import DynibarZQ
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import *
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, save_current_code
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset
import imageio
from ibrnet.data_loaders.flow_utils import flow_to_image
from torch_efficient_distloss import eff_distloss_native
from utils import compute_space_carving_loss

DEBUG = False
USE_OCCU_WEIGHTS = True

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):
    torch.cuda.set_device(args.local_rank)
    # args.expname = args.expname + \
    # 'occ_weights_mode-%d'%(args.occ_weights_mode) + \
    # '_w-disp-%.4f'%(args.w_disp) + \
    # '_anneal_cycle-%d'%args.anneal_cycle + \
    # '_sfm_reg-%d'%args.sfm_reg + \
    # '_w-flow-%.4f'%(args.w_flow) + \
    # '_w-distortion-%.4f'%args.w_distortion + \
    # '_w-se-%.2f-%.4f'%(args.skewness, args.w_skew_entropy)

    args.expname = args.expname 

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, 'out', args.expname)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, 'config.txt')
        # if not os.path.isfile(f):
        shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args) 
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)
    num_frames = args.num_frames = train_dataset.num_frames
    args.lrate_decay_steps = args.num_frames * args.init_decay_epoch


    # Create IBRNet model
    model = DynibarZQ(args, 
        load_opt=True, 
        load_scheduler=True)
    # create projector
    projector = Projector(device=device)

    # Create criterion
    rgb_criterion = Criterion()
    tb_dir = os.path.join(args.rootdir, 'logs/', args.expname)
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}

    global_step = model.start_step 
    start_epoch = global_step // num_frames
    decay_rate = args.decay_rate

    for epoch in range(start_epoch, int(10**5)):
        if global_step > model.start_step + args.n_iters + 1:
            break

        train_dataset.set_global_step(global_step)
        train_dataset.set_epoch(epoch)
        print('====================================== ', epoch)

        for ii, train_data in enumerate(train_loader):
            time0 = time.time()
            ref_time_embedding = train_data['ref_time'].to(device)
            anchor_time_embedding = train_data['anchor_time'].to(device)

            nearest_pose_ids = train_data['nearest_pose_ids'].squeeze().tolist()

            anchor_nearest_pose_ids = train_data['anchor_nearest_pose_ids'].squeeze().tolist()

            ref_frame_idx = int(train_data['id'].item())
            anchor_frame_idx = int(train_data['anchor_id'].item())

            ref_time_offset = [int(near_idx - ref_frame_idx) for near_idx in nearest_pose_ids[:6]] 
            anchor_time_offset = [int(near_idx - anchor_frame_idx) for near_idx in anchor_nearest_pose_ids] 

            if DEBUG:
                print('Logging current training view...')

                print("ref_frame_idx ", ref_frame_idx)
                print("ref_time_embedding ", ref_time_embedding)
                print("ref_time_offset ", ref_time_offset)

                tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device)
                H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                gt_disp = tmp_ray_train_sampler.disp.reshape(H, W, 1)
                log_view_to_tb(writer, global_step, args, 
                               model, tmp_ray_train_sampler, projector,
                               gt_img, gt_disp,
                               frame_idx=(ref_frame_idx, anchor_frame_idx),
                               time_embedding=(ref_time_embedding, anchor_time_embedding),
                               time_offset=(ref_time_offset, anchor_time_offset),
                               render_stride=1, prefix='train/')
                print("we are good")
                sys.exit()

            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(1.0 * args.N_rand)

            ray_batch = ray_sampler.random_sample(N_rand,
                                                  sample_mode=args.sample_mode,
                                                  center_ratio=0.8,
                                                  )
            
            num_dy_views = ray_batch['src_rgbs'].shape[1]
            num_anchor_views = ray_batch['anchor_src_rgbs'].shape[1]

            # assert num_st_views == 11
            cb_src_rgbs = torch.cat([ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2),
                                     ray_batch['anchor_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)],
                                     dim=0)

            cb_featmaps_1, _ = model.feature_net(cb_src_rgbs)
            ref_featmaps, anchor_featmaps = cb_featmaps_1[0:num_dy_views], cb_featmaps_1[num_dy_views:]

            static_src_rgbs = ray_batch['static_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
            _, static_featmaps = model.feature_net(static_src_rgbs)

            ret = render_rays(frame_idx=(ref_frame_idx, anchor_frame_idx),
                              time_embedding=(ref_time_embedding, anchor_time_embedding),
                              time_offset=(ref_time_offset, anchor_time_offset),
                              ray_batch=ray_batch,
                              model=model,
                              projector=projector,
                              featmaps=(ref_featmaps, anchor_featmaps, static_featmaps),
                              N_samples=args.N_samples,
                              args=args,
                              inv_uniform=args.inv_uniform,
                              N_importance=args.N_importance,
                              det=args.det,
                              white_bkgd=args.white_bkgd)

            # # compute loss
            model.optimizer.zero_grad()
            # # RGB loss
            rgb_loss = rgb_criterion(ret['outputs_coarse_ref'], ray_batch)
            rgb_loss += compute_temporal_rgb_loss(ret['outputs_coarse_anchor'], ray_batch)

            divsor = epoch // args.init_decay_epoch

            # RGB loss for dynamic component only
            dynamic_rgb_decay_rate = 5.
            rgb_loss += rgb_criterion(ret['outputs_coarse_ref_dy'], ray_batch, ray_batch['motion_mask']) / ( (dynamic_rgb_decay_rate) ** divsor) 
            rgb_loss += compute_temporal_rgb_loss(ret['outputs_coarse_anchor_dy'], ray_batch, ray_batch['motion_mask']) / ( (dynamic_rgb_decay_rate) ** divsor) 

            # disparity loss
            w_disp = args.w_disp / (decay_rate ** divsor) 
            pred_disp = 1. / torch.clamp(ret['outputs_coarse_ref']["depth"], min=1e-2)

            gt_disp = ray_batch['disp']
            pred_mask = ret['outputs_coarse_ref']["mask"]
            disp_loss = w_disp * torch.sum(torch.abs(pred_disp - gt_disp) * pred_mask) / (torch.sum(pred_mask) + 1e-8)

            # SCADE loss
            ### SCADE ###
            pred_samples = ret['outputs_coarse_ref']["scade_samples"]
            gt_samples = 1.0 / torch.clamp(
                ray_batch['disp'], min=1e-2
            )

            # [num_hypothesis, N_batch, 1]
            gt_samples = gt_samples.unsqueeze(0)
            gt_samples = gt_samples.unsqueeze(-1)

            space_carving_loss = compute_space_carving_loss(pred_samples, gt_samples, is_joint=args.is_joint, mask=pred_mask)
            
            space_carving_loss = args.w_scade * space_carving_loss
            #############


            # # flow loss
            w_flow = args.w_flow / (decay_rate ** divsor)
            flow_mask = pred_mask[None, :, None] * ray_batch['masks']
   
            # forward flow
            flow_loss = w_flow * compute_flow_loss(ret['outputs_coarse_ref']["render_flows"], 
                                                   ray_batch['flows'], 
                                                   flow_mask)
    
            # traj consistency loss
            if args.anneal_cycle:
                w_cycle = min(0.5, args.w_cycle + divsor * 0.1)
            else:
                w_cycle = args.w_cycle

            pts_traj_anchor = ret['outputs_coarse_anchor']['pts_traj_anchor']
            pts_traj_ref = ret['outputs_coarse_anchor']['pts_traj_ref']

            occ_weights = ret['outputs_coarse_anchor']['occ_weights'][None, ..., None].repeat(pts_traj_anchor.shape[0], 
                                                                                              1, 1, 
                                                                                              pts_traj_anchor.shape[-1])
            cycle_loss = w_cycle * torch.sum(torch.abs((pts_traj_ref - pts_traj_anchor)) * occ_weights) / (torch.sum(occ_weights) + 1e-8)

            # traj reg loss
            # minimal scene flow loss
            reg_loss = args.w_sf_zero * torch.mean(torch.abs((ret['outputs_coarse_anchor']['sf_seq'])))
            # temporal smooth loss
            reg_loss += args.w_sf_sm * 0.5 * torch.mean(torch.pow(ret['outputs_coarse_anchor']['sf_seq'][:-1] \
                - ret['outputs_coarse_anchor']['sf_seq'][1:], 2))
            # spatial smooth loss
            reg_loss += args.w_sf_sm * torch.mean(torch.abs(ret['outputs_coarse_anchor']['sf_seq'][:, :, 1:, :] \
                - ret['outputs_coarse_anchor']['sf_seq'][:, :, :-1, :]))

            # skew entropy loss
            render_weights_dy = torch.sum(ret['outputs_coarse_ref']['weights_dy'], dim=-1)
            render_weights_st = torch.sum(ret['outputs_coarse_ref']['weights_st'], dim=-1)
            weights_ratio = render_weights_dy / torch.clamp(render_weights_dy + render_weights_st, min=1e-9) 
            weights_ratio = weights_ratio ** args.skewness

            skew_entropy_loss = -(weights_ratio * torch.log(weights_ratio + 1e-9) + (1. - weights_ratio) * torch.log(1. - weights_ratio + 1e-9))
            skew_entropy_loss = args.w_skew_entropy * torch.mean(skew_entropy_loss)
            
            # distortion loss used in mip-360
            s_vals = ret['outputs_coarse_ref']['s_vals']
            mid_dist = (s_vals[:, 1:] + s_vals[:, :-1]) * 0.5
            interval = (s_vals[:, 1:] - s_vals[:, :-1])

            w_distortion = args.w_distortion
            distortion_loss = w_distortion * eff_distloss_native(ret['outputs_coarse_ref']['weights'][:, :-1], 
                                                                 mid_dist, 
                                                                 interval)
            # scene flow reg
            if args.sfm_reg:
                static_sfm_mask = 1. - ray_batch['static_mask'].float()
                static_sfm_mask *= ret['outputs_coarse_ref']['mask'].float()
                sfm_loss = compute_rgb_loss(ret['outputs_coarse_ref']['rgb_static'], ray_batch, static_sfm_mask) #/ (decay_rate ** divsor)
            else:
                sfm_loss = torch.zeros(1).cuda()

            loss = cycle_loss + flow_loss + disp_loss + rgb_loss + \
                   reg_loss + skew_entropy_loss + \
                   distortion_loss + sfm_loss + space_carving_loss

            loss.backward()
            
            scalars_to_log['loss'] = loss.item()
            scalars_to_log['flow_loss'] = flow_loss.item()
            scalars_to_log['disp_loss'] = disp_loss.item()
            scalars_to_log['rgb_loss'] = rgb_loss.item()
            scalars_to_log['distortion_loss'] = distortion_loss.item()
            scalars_to_log['skew_entropy_loss'] = skew_entropy_loss.item()
            scalars_to_log['sfm_loss'] = sfm_loss.item()
            scalars_to_log['space_carving_loss'] = space_carving_loss.item()
            # scalars_to_log['zero_loss'] = zero_loss.item()

            model.optimizer.step()
            
            if model.scheduler.get_last_lr()[0] > 5e-7:
                model.scheduler.step()

            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            if args.local_rank == 0 and global_step % 10 == 0:
                print("expname ", args.expname)
                print("w_disp ", w_disp, " w_flow ", w_flow)
                print("disp_loss ", disp_loss.item(), " flow_loss ", flow_loss.item() , ' rgb_loss ', rgb_loss.item())
                print("cycle_loss ", cycle_loss.item(), " reg_loss ", reg_loss.item(), ' skew_entropy_loss ', skew_entropy_loss.item())
                print('distortion_loss ', distortion_loss.item(), "sfm_loss ", sfm_loss.item())
                print('w_scade ', args.w_scade, "space_carving_loss ", space_carving_loss.item())
                print("epoch %d global_step %d"%(epoch, global_step), " dt optimization ", dt)

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0:
                    # write mse and psnr stats
                    mse_error = img2mse(ret['outputs_coarse_ref']['rgb'], ray_batch['rgb']).item()
                    scalars_to_log['train/coarse-loss'] = mse_error
                    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
                    if ret['outputs_fine'] is not None:
                        mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                        scalars_to_log['train/fine-loss'] = mse_error
                        scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    print(logstr)
                    print('each iter time {:.05f} seconds'.format(dt))

                if global_step % args.i_weights == 0:
                    print('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_latest.pth'.format(global_step))
                    model.save_model(fpath, global_step)

                if global_step % args.i_img == 0:
                    print('Logging current training view...')
                    tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device)
                    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                    gt_disp = tmp_ray_train_sampler.disp.reshape(H, W, 1)
                    log_view_to_tb(writer, global_step, args, 
                                   model, tmp_ray_train_sampler, projector,
                                   gt_img, gt_disp,
                                   frame_idx=(ref_frame_idx, anchor_frame_idx),
                                   time_embedding=(ref_time_embedding, anchor_time_embedding),
                                   time_offset=(ref_time_offset, anchor_time_offset),
                                   render_stride=1, 
                                   prefix='train/')

                    torch.cuda.empty_cache()

            global_step += 1


def log_view_to_tb(writer, global_step, args, 
                   model, ray_sampler, projector, 
                   gt_img, gt_disp,
                   frame_idx, time_embedding, time_offset,
                   render_stride=1, prefix=''):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            num_dy_views = 6
            num_anchor_views = ray_batch['anchor_src_rgbs'].shape[1]
            cb_src_rgbs = torch.cat([ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2),
                                     ray_batch['anchor_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)],
                                     dim=0)

            cb_featmaps_1, _ = model.feature_net(cb_src_rgbs)
            ref_featmaps, anchor_featmaps = cb_featmaps_1[0:num_dy_views], cb_featmaps_1[num_dy_views:]

            static_src_rgbs = ray_batch['static_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
            _, static_featmaps = model.feature_net(static_src_rgbs)

            featmaps = (ref_featmaps, anchor_featmaps, static_featmaps)
        else:
            featmaps = [None, None]

        ret = render_single_image(frame_idx=frame_idx,
                                  time_embedding=time_embedding,
                                  time_offset=time_offset,
                                  ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  projector=projector,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  args=args,
                                  inv_uniform=args.inv_uniform,
                                  det=True,
                                  N_importance=args.N_importance,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps)

        if DEBUG:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import sys

            print("frame_idx ", frame_idx[0])
            print("time_embedding ", time_embedding[0])
            print("time_offset ", time_offset[0])
            gt_flows = ray_batch['flows'].reshape(ray_batch['flows'].shape[0], gt_img.shape[0], gt_img.shape[1], 2)

            # for ii in range(gt_flows.shape[0]):
                # print(ii)
                # combined_flow = np.concatenate([flow_to_image(gt_flows[ii].cpu().numpy())/255., flow_to_image(ret['outputs_coarse_ref']['render_flows'][ii].cpu().numpy())/255.], axis=1)
                # imageio.imwrite('viz_flow_%3d.png'%ii, combined_flow)

            ref_rgb_render = ret['outputs_coarse_ref']['rgb'].detach().cpu().numpy()
            ref_depth_render = ret['outputs_coarse_ref']['depth'].detach().cpu().numpy()

            imageio.imwrite('debug_rgb.png', np.clip(ref_rgb_render, 0., 1.))

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(ref_rgb_render)
            plt.subplot(1, 2, 2)
            plt.imshow(ref_depth_render, cmap='jet') 
            # plt.subplot(2, 3, 3)
            # plt.imshow(gt_disp[..., 0].cpu().numpy(), cmap='gray')
            # plt.subplot(2, 3, 4)
            # plt.imshow(ret['outputs_coarse']["depth"].cpu().numpy(), cmap='gray')
            plt.tight_layout()
            plt.savefig('rgb-depth_viz.png')
            plt.close()

            sys.exit()
        # imageio.imwrite('occ_weight_map_threshold.png', np.float32(occ_weight_map.numpy() > 0.999))
        # imageio.imwrite('occ_weight_map_viz.png', np.float32(occ_weight_map_viz.numpy().transpose(1, 2, 0)))

        # exp_sf = ret['outputs_coarse_anchor']['exp_sf'].detach().cpu()
        # exp_sf_mag = torch.norm(exp_sf, dim=-1) 

        # print("exp_sf_mag ", exp_sf_mag.shape, torch.min(torch.abs(exp_sf_mag)), torch.max(torch.abs(exp_sf_mag)))
        # exp_sf_mag = (exp_sf_mag) / torch.max(exp_sf_mag)
        # imageio.imwrite('exp_sf_map.png', (exp_sf_mag).float().numpy())

        # occ_weight_map = ret['outputs_coarse_anchor']['occ_weight_map'].detach().cpu()
        # print("occ_weight_map ", np.min(occ_weight_map.numpy()))

        # occ_weight_map_viz = img_HWC2CHW(colorize(occ_weight_map, cmap_name='gray', append_cbar=False))
        # print("occ_weight_map_viz ", occ_weight_map_viz.shape)
        # imageio.imwrite('occ_weight_map_threshold.png', np.float32(occ_weight_map.numpy() > 0.999))
        # imageio.imwrite('occ_weight_map_viz.png', np.float32(occ_weight_map_viz.numpy().transpose(1, 2, 0)))
        # sys.exit()


    # average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    # if args.render_stride != 1:
    #     gt_img = gt_img[::render_stride, ::render_stride]
    #     # average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    ref_rgb_pred = img_HWC2CHW(ret['outputs_coarse_ref']['rgb'].detach().cpu())
    static_rgb_pred = img_HWC2CHW(ret['outputs_coarse_ref']['rgb_static'].detach().cpu())
    dy_rgb_pred = img_HWC2CHW(ret['outputs_coarse_ref']['rgb_dy'].detach().cpu())

    anchor_rgb_pred = img_HWC2CHW(ret['outputs_coarse_anchor']['rgb'].detach().cpu())

    gt_flows = ray_batch['flows'].reshape(ray_batch['flows'].shape[0], gt_img.shape[0], gt_img.shape[1], 2)

    exp_sf = ret['outputs_coarse_ref']['exp_sf'].detach().cpu()
    exp_sf_mag = torch.norm(exp_sf, dim=-1) 


    if False:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import sys

        # print(torch.median(torch.norm(ret['outputs_coarse']['render_fwd_flows'][0].cpu() - gt_fwd_flows[0].cpu(), dim=-1)))
        # print(torch.median(torch.norm(ret['outputs_coarse']['render_bwd_flows'][0].cpu() - gt_bwd_flows[0].cpu(), dim=-1)))

        # print(torch.median(torch.norm(ret['outputs_coarse']['render_fwd_flows'][1].cpu() - gt_fwd_flows[1].cpu(), dim=-1)))
        # print(torch.median(torch.norm(ret['outputs_coarse']['render_bwd_flows'][1].cpu() - gt_bwd_flows[1].cpu(), dim=-1)))

        # print(torch.median(torch.norm(ret['outputs_coarse']['render_fwd_flows'][2].cpu() - gt_fwd_flows[2].cpu(), dim=-1)))
        # print(torch.median(torch.norm(ret['outputs_coarse']['render_bwd_flows'][2].cpu() - gt_bwd_flows[2].cpu(), dim=-1)))

        # plt.figure(figsize=(19, 6))
        # plt.subplot(2, 2, 1)
        # plt.imshow(gt_img.cpu().numpy())
        # plt.subplot(2, 2, 2)
        # plt.imshow(ret['outputs_coarse']['rgb'].cpu().numpy()) 
        # plt.subplot(2, 2, 3)
        # plt.imshow(gt_disp[..., 0].cpu().numpy(), cmap='gray')
        # plt.subplot(2, 2, 4)
        # plt.imshow(ret['outputs_coarse']["depth"].cpu().numpy(), cmap='gray')
        # plt.tight_layout()
        # plt.savefig('render_depth.png')
        # plt.close()

        # plt.figure(figsize=(19, 6))
        # plt.subplot(2, 3, 1)
        # plt.imshow(flow_to_image(ret['outputs_coarse']['render_fwd_flows'][0].cpu().numpy())/255.)
        # plt.subplot(2, 3, 4)
        # plt.imshow(flow_to_image(ret['outputs_coarse']['render_bwd_flows'][0].cpu().numpy())/255.)

        # plt.subplot(2, 3, 2)
        # plt.imshow(flow_to_image(ret['outputs_coarse']['render_fwd_flows'][1].cpu().numpy())/255.)
        # plt.subplot(2, 3, 5)
        # plt.imshow(flow_to_image(ret['outputs_coarse']['render_bwd_flows'][1].cpu().numpy())/255.)

        # plt.subplot(2, 3, 3)
        # plt.imshow(flow_to_image(ret['outputs_coarse']['render_fwd_flows'][2].cpu().numpy())/255.)
        # plt.subplot(2, 3, 6)
        # plt.imshow(flow_to_image(ret['outputs_coarse']['render_bwd_flows'][2].cpu().numpy())/255.)

        # plt.tight_layout()
        # plt.savefig('render_flow.png')

        plt.figure(figsize=(19, 6))
        plt.subplot(2, 3, 1)
        plt.imshow(flow_to_image(gt_flows[0].cpu().numpy())/255.)
        plt.subplot(2, 3, 4)
        plt.imshow(flow_to_image(gt_flows[1].cpu().numpy())/255.)

        plt.subplot(2, 3, 2)
        plt.imshow(flow_to_image(gt_flows[2].cpu().numpy())/255.)
        plt.subplot(2, 3, 5)
        plt.imshow(flow_to_image(gt_flows[3].cpu().numpy())/255.)

        plt.subplot(2, 3, 3)
        plt.imshow(flow_to_image(gt_flows[4].cpu().numpy())/255.)
        plt.subplot(2, 3, 6)
        plt.imshow(flow_to_image(gt_flows[5].cpu().numpy())/255.)

        plt.tight_layout()
        plt.savefig('gt_flow.png')
        sys.exit()

    ref_depth_im = ret['outputs_coarse_ref']['depth'].detach().cpu()
    anchor_depth_im = ret['outputs_coarse_anchor']['depth'].detach().cpu()
    occ_weight_map = ret['outputs_coarse_anchor']['occ_weight_map'].detach().cpu()

    # ref_mask = ret['outputs_coarse_ref']['mask'].detach().cpu().view(depth_im.shape)

    # acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()

    # print("depth_im ", depth_im.shape)
    # print("gt_disp ", gt_disp.shape)

    if ret['outputs_fine'] is None:

        writer.add_image(prefix + 'render_rgb_coarse_ref', torch.clamp(ref_rgb_pred, 0., 1.), global_step)
        writer.add_image(prefix + 'render_rgb_coarse_anchor', torch.clamp(anchor_rgb_pred, 0., 1.), global_step)
        writer.add_image(prefix + 'render_rgb_static', torch.clamp(static_rgb_pred, 0., 1.), global_step)
        writer.add_image(prefix + 'render_rgb_dynamic', torch.clamp(dy_rgb_pred, 0., 1.), global_step)
        
        render_depth_ref = img_HWC2CHW(colorize(ref_depth_im, cmap_name='jet', append_cbar=False))
        render_depth_anchor = img_HWC2CHW(colorize(anchor_depth_im, cmap_name='jet', append_cbar=False))

        gt_disp_viz = img_HWC2CHW(colorize(gt_disp[..., 0], cmap_name='jet', append_cbar=False))
        occ_weight_map_viz = img_HWC2CHW(colorize(occ_weight_map, cmap_name='gray', append_cbar=False))
        exp_sf_mag = img_HWC2CHW(colorize(exp_sf_mag, cmap_name='gray', append_cbar=False))

        writer.add_image(prefix + 'render_depth_coarse', render_depth_ref, global_step)
        writer.add_image(prefix + 'anchor_depth_coarse', render_depth_anchor, global_step)
        writer.add_image(prefix + 'occ_weight_map', occ_weight_map_viz, global_step)
        # writer.add_image(prefix + 'occ_weight_map_threshold', (occ_weight_map > 0.9).float(), global_step, dataformats='HW')
        writer.add_image(prefix + 'exp_sf_mag', exp_sf_mag, global_step)

        writer.add_image(prefix + 'gt_disp_coarse', gt_disp_viz, global_step)
        writer.add_image(prefix + 'gt_rgb_coarse', rgb_gt, global_step)

        # write flow
        rd_flow_stack = []
        gt_flow_stack = []
        for ii in range(gt_flows.shape[0]):
            rd_flow_stack.append(torch.Tensor(flow_to_image(ret['outputs_coarse_ref']['render_flows'][ii].cpu().numpy())/255.))
            gt_flow_stack.append(torch.Tensor(flow_to_image(gt_flows[ii].cpu().numpy())/255.))

        rd_flow_stack = torch.stack(rd_flow_stack, dim=0)
        gt_flow_stack = torch.stack(gt_flow_stack, dim=0)

        writer.add_images(prefix + 'rd_flow_stack', rd_flow_stack, global_step=global_step,  dataformats='NHWC')
        writer.add_images(prefix + 'gt_flow_stack', gt_flow_stack, global_step=global_step,  dataformats='NHWC')

    else:
        rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    model.switch_to_train()
    return 

if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)
