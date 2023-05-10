'''
May 1, 2023
Updated script with depth supervision
'''
import os
import shutil
import subprocess
import math
import time
import datetime
from argparse import Namespace

import configargparse
from skimage.metrics import structural_similarity
from lpips import LPIPS
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from model import NeRF, get_embedder, get_rays, precompute_quadratic_samples, sample_pdf, img2mse, mse2psnr, to8b, \
    compute_depth_loss, select_coordinates, to16b, resnet18_skip, sample_pdf_reformulation, compute_space_carving_loss, get_space_carving_idx, compute_space_carving_loss_corrected, \
    sample_pdf_return_u, get_space_carving_idx_corrected, sample_pdf_joint_return_u, sample_pdf_reformulation_joint, sample_pdf_reformulation_return_u, sample_pdf_reformulation_joint_return_u, \
    Scale_Gradient_PDF
from data import create_random_subsets, load_scene, convert_depth_completion_scaling_to_m, \
    convert_m_to_depth_completion_scaling, get_pretrained_normalize, resize_sparse_depth, load_scene_llff, load_scene_blender, load_scene_mika, load_scene_blender_depth, load_scene_blender2_depth
from train_utils import MeanTracker, update_learning_rate
from metric import compute_rmse

import imageio
from natsort import natsorted 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, embedded_cam, fn, embed_fn, embeddirs_fn, bb_center, bb_scale, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    inputs_flat = (inputs_flat - bb_center) * bb_scale ### --> does this make sense for inward facing cameras?

    embedded = embed_fn(inputs_flat) # samples * rays, multires * 2 * 3 + 3

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs, embedded_cam.unsqueeze(0).expand(embedded_dirs.shape[0], embedded_cam.shape[0])], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, use_viewdirs=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], use_viewdirs, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, intrinsic, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., with_5_9=False, use_viewdirs=False, c2w_staticcam=None, 
                  rays_depth=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      with_5_9: render with aspect ratio 5.33:9 (one third of 16:9)
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, intrinsic, c2w)
        if with_5_9:
            W_before = W
            W = int(H / 9. * 16. / 3.)
            if W % 2 != 0:
                W = W - 1
            start = (W_before - W) // 2
            rays_o = rays_o[:, start:start + W, :]
            rays_d = rays_d[:, start:start + W, :]
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        rays_o, rays_d, rays_depth = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, intrinsic, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_hyp(H, W, intrinsic, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., with_5_9=False, use_viewdirs=False, c2w_staticcam=None, 
                  rays_depth=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      with_5_9: render with aspect ratio 5.33:9 (one third of 16:9)
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, intrinsic, c2w)
        if with_5_9:
            W_before = W
            W = int(H / 9. * 16. / 3.)
            if W % 2 != 0:
                W = W - 1
            start = (W_before - W) // 2
            rays_o = rays_o[:, start:start + W, :]
            rays_d = rays_d[:, start:start + W, :]
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        rays_o, rays_d, rays_depth = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, intrinsic, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # print(rays_o.shape)
    # print(rays_d.shape)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    # print(near)
    # print(far)
    # print(use_viewdirs)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # print(rays.shape)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

### range from [mean-3*sd, mean+3*sd]
def precompute_depth_sampling(depth):
    depth_min = (depth[:, 0] - 3. * depth[:, 1])
    depth_max = depth[:, 0] + 3. * depth[:, 1]
    return torch.stack((depth[:, 0], depth_min, depth_max), -1)

# def render_video(poses, H, W, intrinsics, filename, args, render_kwargs_test, fps=25):
#     video_dir = os.path.join(args.ckpt_dir, args.expname, 'video_' + filename)
#     if os.path.exists(video_dir):
#         shutil.rmtree(video_dir)
#     os.makedirs(video_dir, exist_ok=True)
#     depth_scale = render_kwargs_test["far"]
#     max_depth_in_video = 0
#     for img_idx in range(0, len(poses), 3):
#     # for img_idx in range(200):
#         pose = poses[img_idx, :3,:4]
#         intrinsic = intrinsics[img_idx, :]
#         with torch.no_grad():
#             if args.input_ch_cam > 0:
#                 render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
#             # render video in 16:9 with one third rgb, one third depth and one third depth standard deviation
#             rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, with_5_9=True, **render_kwargs_test)
#             rgb_cpu_numpy_8b = to8b(rgb.cpu().numpy())
#             video_frame = cv2.cvtColor(rgb_cpu_numpy_8b, cv2.COLOR_RGB2BGR)
#             max_depth_in_video = max(max_depth_in_video, extras['depth_map'].max())
#             depth_frame = cv2.applyColorMap(to8b((extras['depth_map'] / depth_scale).cpu().numpy()), cv2.COLORMAP_TURBO)
#             video_frame = np.concatenate((video_frame, depth_frame), 1)
#             depth_var = ((extras['z_vals'] - extras['depth_map'].unsqueeze(-1)).pow(2) * extras['weights']).sum(-1)
#             depth_std = depth_var.clamp(0., 1.).sqrt()
#             video_frame = np.concatenate((video_frame, cv2.applyColorMap(to8b(depth_std.cpu().numpy()), cv2.COLORMAP_VIRIDIS)), 1)
#             cv2.imwrite(os.path.join(video_dir, str(img_idx) + '.jpg'), video_frame)

#     video_file = os.path.join(args.ckpt_dir, args.expname, filename + '.mp4')
#     subprocess.call(["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(video_dir, "%d.jpg"), "-c:v", "libx264", "-profile:v", "high", "-crf", str(fps), video_file])
#     print("Maximal depth in video: {}".format(max_depth_in_video))

def render_video(poses, H, W, intrinsics, filename, args, render_kwargs_test, fps=25):
    video_dir = os.path.join(args.ckpt_dir, args.expname, 'video_demo2_' + filename)
    video_depth_dir = os.path.join(args.ckpt_dir, args.expname, 'video_demo2_depth_' + filename)
    video_depth_colored_dir = os.path.join(args.ckpt_dir, args.expname, 'video_demo2_depth_colored' + filename)

    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    if os.path.exists(video_depth_dir):
        shutil.rmtree(video_depth_dir)
    if os.path.exists(video_depth_colored_dir):
        shutil.rmtree(video_depth_colored_dir)        

    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(video_depth_dir, exist_ok=True)
    os.makedirs(video_depth_colored_dir, exist_ok=True)

    depth_scale = render_kwargs_test["far"]
    max_depth_in_video = 0

    # start_idx = int(len(poses)*(24./35.))
    # end_idx = int(len(poses)*(32./35.))+1

    start_idx = 0
    end_idx = len(poses)

    for img_idx in range(start_idx, end_idx):
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]
        with torch.no_grad():
            if args.input_ch_cam > 0:
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
            # render video in 16:9 with one third rgb, one third depth and one third depth standard deviation
            rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, with_5_9=False, **render_kwargs_test)
            rgb_cpu_numpy_8b = to8b(rgb.cpu().numpy())
            video_frame = cv2.cvtColor(rgb_cpu_numpy_8b, cv2.COLOR_RGB2BGR)

            max_depth_in_video = max(max_depth_in_video, extras['depth_map'].max())
            depth_colored_frame = cv2.applyColorMap(to8b((extras['depth_map'] / depth_scale).cpu().numpy()), cv2.COLORMAP_TURBO)
            depth = (extras['depth_map']).cpu().numpy()*1000.
            depth = (depth).astype(np.uint16)

            # video_frame = np.concatenate((video_frame, depth_frame), 1)
            # depth_var = ((extras['z_vals'] - extras['depth_map'].unsqueeze(-1)).pow(2) * extras['weights']).sum(-1)
            # depth_std = depth_var.clamp(0., 1.).sqrt()
            # video_frame = np.concatenate((video_frame, cv2.applyColorMap(to8b(depth_std.cpu().numpy()), cv2.COLORMAP_VIRIDIS)), 1)

            cv2.imwrite(os.path.join(video_dir, str(img_idx) + '.png'), video_frame)        
            cv2.imwrite(os.path.join(video_depth_dir, str(img_idx) + '.png'), depth)
            cv2.imwrite(os.path.join(video_depth_colored_dir, str(img_idx) + '.png'), depth_colored_frame)


    video_file = os.path.join(args.ckpt_dir, args.expname, filename + '.mp4')

    print(video_dir)
    imgs = os.listdir(video_dir)
    imgs = natsorted(imgs)
    print(imgs)

    imageio.mimsave(video_file,
                    [imageio.imread(os.path.join(video_dir, img)) for img in imgs],
                    fps=10, macro_block_size=1)
    print("Done.")


def optimize_camera_embedding(image, pose, H, W, intrinsic, args, render_kwargs_test):
    render_kwargs_test["embedded_cam"] = torch.zeros(args.input_ch_cam, requires_grad=True).to(device)
    optimizer = torch.optim.Adam(params=(render_kwargs_test["embedded_cam"],), lr=5e-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3, verbose=True)
    half_W = W
    print(" - Optimize camera embedding")
    max_psnr = 0
    best_embedded_cam = torch.zeros(args.input_ch_cam).to(device)
    # make batches
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, half_W - 1, half_W), indexing='ij'), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2]).long()
    assert(coords[:, 1].max() < half_W)
    batches = create_random_subsets(range(len(coords)), 2 * args.N_rand, device=device)
    # make rays
    rays_o, rays_d = get_rays(H, half_W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    start_time = time.time()
    for i in range(100):
        sum_img_loss = torch.zeros(1)
        optimizer.zero_grad()
        for b in batches:
            curr_coords = coords[b]
            curr_rays_o = rays_o[curr_coords[:, 0], curr_coords[:, 1]]  # (N_rand, 3)
            curr_rays_d = rays_d[curr_coords[:, 0], curr_coords[:, 1]]  # (N_rand, 3)
            target_s = image[curr_coords[:, 0], curr_coords[:, 1]]
            batch_rays = torch.stack([curr_rays_o, curr_rays_d], 0)
            rgb, _, _, _ = render(H, half_W, None, chunk=args.chunk, rays=batch_rays, verbose=i < 10, **render_kwargs_test)
            img_loss = img2mse(rgb, target_s)
            img_loss.backward()
            sum_img_loss += img_loss
        optimizer.step()
        psnr = mse2psnr(sum_img_loss / len(batches))
        lr_scheduler.step(psnr)
        if psnr > max_psnr:
            max_psnr = psnr
            best_embedded_cam = render_kwargs_test["embedded_cam"].detach().clone()
            print("Step {}: PSNR: {} ({:.2f}min)".format(i, psnr, (time.time() - start_time) / 60))
    render_kwargs_test["embedded_cam"] = best_embedded_cam

def render_images_with_metrics(count, indices, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, \
    embedcam_fn=None, with_test_time_optimization=False):
    far = render_kwargs_test['far']

    if count is None:
        # take all images in order
        count = len(indices)
        img_i = indices
    else:
        # take random images
        if count > len(indices):
            count = len(indices)
        img_i = np.random.choice(indices, size=count, replace=False)

    rgbs_res = torch.empty(count, 3, H, W)
    rgbs0_res = torch.empty(count, 3, H, W)
    target_rgbs_res = torch.empty(count, 3, H, W)
    depths_res = torch.empty(count, 1, H, W)
    depths0_res = torch.empty(count, 1, H, W)
    target_depths_res = torch.empty(count, 1, H, W)
    target_valid_depths_res = torch.empty(count, 1, H, W, dtype=bool)
    
    mean_metrics = MeanTracker()
    mean_depth_metrics = MeanTracker() # track separately since they are not always available
    for n, img_idx in enumerate(img_i):
        print("Render image {}/{}".format(n + 1, count), end="")
        target = images[img_idx]

        target_depth = depths[img_idx]
        target_valid_depth = valid_depths[img_idx]

        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]

        if args.input_ch_cam > 0:
            if embedcam_fn is None:
                # use zero embedding at test time or optimize for the latent code
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
                if with_test_time_optimization:
                    optimize_camera_embedding(target, pose, H, W, intrinsic, args, render_kwargs_test)
                    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_latent_codes_" + args.scene_id)
                    os.makedirs(result_dir, exist_ok=True)
                    np.savetxt(os.path.join(result_dir, str(img_idx) + ".txt"), render_kwargs_test["embedded_cam"].cpu().numpy())
            else:
                render_kwargs_test["embedded_cam"] = embedcam_fn(torch.tensor(img_idx, device=device))
        
        with torch.no_grad():
            # rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, **render_kwargs_test)
            # print(render_kwargs_test)
            rgb, _, _, extras = render(H, W, intrinsic, chunk=args.chunk, c2w=pose, **render_kwargs_test)
            
            # compute depth rmse
            depth_rmse = compute_rmse(extras['depth_map'][target_valid_depth], target_depth[:, :, 0][target_valid_depth])
            if not torch.isnan(depth_rmse):
                depth_metrics = {"depth_rmse" : depth_rmse.item()}
                mean_depth_metrics.add(depth_metrics)

            # compute color metrics
            img_loss = img2mse(rgb, target)
            psnr = mse2psnr(img_loss)
            print("PSNR: {}".format(psnr))
            rgb = torch.clamp(rgb, 0, 1)
            ssim = structural_similarity(rgb.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)
            lpips = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
            
            # store result
            rgbs_res[n] = rgb.clamp(0., 1.).permute(2, 0, 1).cpu()
            target_rgbs_res[n] = target.permute(2, 0, 1).cpu()
            depths_res[n] = (extras['depth_map'] / far).unsqueeze(0).cpu()
            target_depths_res[n] = (target_depth[:, :, 0] / far).unsqueeze(0).cpu()
            target_valid_depths_res[n] = target_valid_depth.unsqueeze(0).cpu()

            print("Raw depth")
            print(extras['depth_map'])
            print()
            print(target_depth)

            print()
            print("outputted depth")
            print(depths_res[n])
            print()
            print(target_depths_res[n])
            exit()

            metrics = {"img_loss" : img_loss.item(), "psnr" : psnr.item(), "ssim" : ssim, "lpips" : lpips[0, 0, 0],}
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target)
                psnr0 = mse2psnr(img_loss0)
                depths0_res[n] = (extras['depth0'] / far).unsqueeze(0).cpu()
                rgbs0_res[n] = torch.clamp(extras['rgb0'], 0, 1).permute(2, 0, 1).cpu()
                metrics.update({"img_loss0" : img_loss0.item(), "psnr0" : psnr0.item()})
            mean_metrics.add(metrics)
    
    res = { "rgbs" :  rgbs_res, "target_rgbs" : target_rgbs_res, "depths" : depths_res, "target_depths" : target_depths_res, \
        "target_valid_depths" : target_valid_depths_res}
    if 'rgb0' in extras:
        res.update({"rgbs0" : rgbs0_res, "depths0" : depths0_res,})
    all_mean_metrics = MeanTracker()
    all_mean_metrics.add({**mean_metrics.as_dict(), **mean_depth_metrics.as_dict()})
    return all_mean_metrics, res

def write_images_with_metrics(images, mean_metrics, far, args, with_test_time_optimization=False, test_samples=False):
    
    if not test_samples:
        result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_" + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id)
    else:
        result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_samples" + ("with_optimization_" if with_test_time_optimization else "") + str(args.N_samples) + "_" + str(args.N_importance) + args.scene_id)

    os.makedirs(result_dir, exist_ok=True)
    for n, (rgb, depth, gt_rgb) in enumerate(zip(images["rgbs"].permute(0, 2, 3, 1).cpu().numpy(), \
            images["depths"].permute(0, 2, 3, 1).cpu().numpy(), images["target_rgbs"].permute(0, 2, 3, 1).cpu().numpy())):

        # write rgb
        # cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".jpg"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".png"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))

        cv2.imwrite(os.path.join(result_dir, str(n) + "_gt" + ".png"), cv2.cvtColor(to8b(gt_rgb), cv2.COLOR_RGB2BGR))

        # write depth
        cv2.imwrite(os.path.join(result_dir, str(n) + "_d" + ".png"), to16b(depth))

    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        mean_metrics.print(f)
    mean_metrics.print()

def load_checkpoint(args):
    path = os.path.join(args.ckpt_dir, args.expname)
    ckpts = [os.path.join(path, f) for f in sorted(os.listdir(path)) if '000.tar' in f]
    print('Found ckpts', ckpts)
    ckpt = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
    return ckpt

def create_nerf(args, scene_render_params):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    if not args.cimle_white_balancing:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)
    else:
        model = NeRF_camlatent_add(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)   


    model = nn.DataParallel(model).to(device)
    grad_vars = list(model.parameters())

    grad_vars = []
    ch_grad_vars = []

    grad_names = []
    ch_grad_names = []


    if not args.cimle_white_balancing:
        for name, param in model.named_parameters():
            grad_vars.append(param)
            grad_names.append(name)
    else:
        for name, param in model.named_parameters():
            if "ch_cam_linear" not in name:
                grad_vars.append(param)
                grad_names.append(name)
            else:
                ch_grad_vars.append(param)
                ch_grad_names.append(name)


    model_fine = None
    if args.N_importance > 0:
        if not args.cimle_white_balancing:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)
        else:
            model_fine = NeRF_camlatent_add(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)            

        model_fine = nn.DataParallel(model_fine).to(device)
        # grad_vars += list(model_fine.parameters())

        if not args.cimle_white_balancing:
            for name, param in model_fine.named_parameters():
                grad_vars.append(param)
                grad_names.append(name)
        else:
            for name, param in model_fine.named_parameters():
                if "ch_cam_linear" not in name:
                    grad_vars.append(param)
                    grad_names.append(name)
                else:
                    ch_grad_vars.append(param)
                    ch_grad_names.append(name)

    network_query_fn = lambda inputs, viewdirs, embedded_cam, network_fn : run_network(inputs, viewdirs, embedded_cam, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                bb_center=args.bb_center,
                                                                bb_scale=args.bb_scale,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    if args.cimle_white_balancing:
        print("Using different learning rate for cimle weights.")
        optimizer.add_param_group({"params": ch_grad_vars, "lr":args.cimle_lrate})
        grad_vars = grad_vars + ch_grad_vars

    start = 0

    ##########################

    # Load checkpoints
    ckpt = load_checkpoint(args)
    if ckpt is not None:
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################
    embedded_cam = torch.tensor((), device=device)
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'embedded_cam' : embedded_cam,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        'white_bkgd' : args.white_bkgd,
        'mode' : args.mode,
        'color_mode': args.color_mode,
        'farcolorfix': args.farcolorfix
    }
    render_kwargs_train.update(scene_render_params)

    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    
    ######################
    ### FIX THIS MIKA  ###
    ######################
    ### To fix current implementation --> perturb at test time too
    # render_kwargs_test['perturb'] = False
    render_kwargs_test['perturb'] = True
    ######################

    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def compute_weights(raw, z_vals, rays_d, noise=0.):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    return weights

### Our reformulation to piecewise linear
def compute_weights_piecewise_linear(raw, z_vals, near, far, rays_d, noise=0., return_tau=False):
    raw2expr = lambda raw, dists: torch.exp(-raw*dists)

    if DEBUG:
        print("In compute piecewise linear.")
        print(raw.shape)
        print(z_vals.shape)
        print(near)
        print(far)
        print(rays_d.shape)
        print()

    ### Concat
    z_vals = torch.cat([near, z_vals, far], -1)

    if DEBUG:
        print("z_vals")
        print(z_vals)
        print()

    ### Make the far plane very far --> force T(last_bin) = 0
    # z_vals[..., -1] = 1e10

    dists = z_vals[...,1:] - z_vals[...,:-1]

    ### Original code
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    if DEBUG:
        print("dist min max")
        print(torch.min(dists))
        print(torch.max(dists))

    # tau = torch.cat([torch.ones((raw.shape[0], 1), device=device)*1e-10, raw[...,3] + noise, raw[...,3][...,-1].unsqueeze(-1)], -1) ### tau(near) = 0, tau(far) = tau(last_sample)
    tau = torch.cat([torch.ones((raw.shape[0], 1), device=device)*1e-10, raw[...,3] + noise, torch.ones((raw.shape[0], 1), device=device)*1e10], -1) ### tau(near) = 0, tau(far) = very big (will hit an opaque surface)

    tau = F.relu(tau) ## Make positive from proof of DS-NeRF

    interval_ave_tau = 0.5 * (tau[...,1:] + tau[...,:-1])
    
    if DEBUG:
        print("Tau min max")
        print(torch.min(tau))
        print(torch.max(tau))

    '''
    Evaluating exp(-0.5 (tau_{i+1}+tau_i) (s_{i+1}-s_i) )
    '''
    expr = raw2expr(interval_ave_tau, dists)  # [N_rays, N_samples+1]
    
    if DEBUG:
        print("expr min max")
        print(torch.min(expr))
        print(torch.max(expr))

    ### Transmittance until s_n
    # T = torch.cumprod(expr, -1)[:, :-1]
    T = torch.cumprod(torch.cat([torch.ones((expr.shape[0], 1), device=device), expr], -1), -1) # [N_rays, N_samples+2], T(near)=1, starts off at 1

    if DEBUG:
        print("T min max")
        print(torch.min(T))
        print(torch.max(T))

    ### Factor to multiply transmittance with
    # factor = (1 - expr)[:, 1:]
    factor = (1 - expr)

    weights = factor * T[:, :-1] # [N_rays, N_samples+1]

    # ### TODO: currently, weights don't sum to 1 --> find a fix to this
    # print(weights)
    # # print()
    # # print(T[..., -1])
    # # print()
    if DEBUG:
        print("==========")
        print("Weights min max")
        print(torch.min(weights))
        print(torch.max(weights))
        print("Weights all sum, min sum, max sum:")
        print(torch.sum(weights, axis=-1))
        print(torch.min(torch.sum(weights, axis=-1)))
        print(torch.max(torch.sum(weights, axis=-1)))
        print("==========")
    # print()
    # print(torch.sum(weights, axis=-1).shape)
    # print(torch.sum(weights, axis=-1))
    # print(torch.max(torch.sum(weights, axis=-1)))
    # print(torch.min(torch.sum(weights, axis=-1)))
    # print()
    # print(weights.shape)
    # print(T.shape)
    # print(tau.shape)
    # exit()

    '''
    We will need to return tau and T for backprop later
    '''
    ### Remember to remove the last value of T(far) is not used
    ### tau(far) is also not used

    if return_tau:
        return weights, tau, T
    else:
        return weights    

def raw2depth(raw, z_vals, near, far, rays_d, mode):

    if mode == "linear":
        weights = compute_weights_piecewise_linear(raw, z_vals, near, far, rays_d)
    elif mode == "constant":
        weights = compute_weights(raw, z_vals, rays_d)

    depth = torch.sum(weights * z_vals, -1)
    std = (((z_vals - depth.unsqueeze(-1)).pow(2) * weights).sum(-1)).sqrt()
    return depth, std

def raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std=0, pytest=False, white_bkgd=False, farcolorfix=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    if mode == "linear":
        weights, tau, T = compute_weights_piecewise_linear(raw, z_vals, near, far, rays_d, noise, return_tau=True)

        if DEBUG:
            print("===============")
            print("In raw2outputs:")
            print("Does nan/inf exist in weights")
            print(torch.isnan(weights).any())
            print(torch.isinf(weights).any())
            print("Does nan exist in per point rgb")
            print(torch.isnan(rgb).any())        
    
        ### Skip the first bin weights [near, s_0]
        if color_mode == "midpoint":

            if farcolorfix:
                # # Make near plane color same as first sample
                # # Make the last sample black --> remove this? this doesnt seem to converge
                # # Make the far plane color also black
                # ## This doesn't seem to converge
                # rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb[: ,:-1, :], torch.zeros((rgb[:, -1].shape), device=device).unsqueeze(1), torch.zeros((rgb[:, -1].shape), device=device).unsqueeze(1)], 1)

                ## Make near plane color same as first sample
                ## Make the far plane color also black
                rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb, torch.zeros((rgb[:, -1].shape), device=device).unsqueeze(1)], 1)

            else:
                rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb, rgb[: ,-1, :].unsqueeze(1)], 1)

            rgb_mid = .5 * (rgb_concat[:, 1:, :] + rgb_concat[:, :-1, :])

            rgb_map = torch.sum(weights[...,None] * rgb_mid, -2)  # [N_rays, 3]

        elif color_mode == "left":

            # ## hack on the last bin doesn't really work
            # if farcolorfix:
            #     ## Make the last sample black
            #     rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb[: ,:-1, :], torch.zeros((rgb[:, -1].shape), device=device).unsqueeze(1)], 1)

            # else:
            #     rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb], 1)

            rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb], 1)
            rgb_map = torch.sum(weights[...,None] * rgb_concat, -2)

        else:
            print("ERROR: Color mode unimplemented, please select left or midpoint.")

        if DEBUG:
            print("Does nan exist in per point rgb_map")
            print(torch.isnan(rgb_map).any())

        ### Piecewise linear means take the midpoint
        z_vals = torch.cat([near, z_vals, far], -1)
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        depth_map = torch.sum(weights * z_vals_mid, -1)

    elif mode == "constant":
        weights = compute_weights(raw, z_vals, rays_d, noise)
        
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)    
        
        tau = None
        T = None    

    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if DEBUG:
        print("Does nan exist in per point rgb_map")
        print(torch.isnan(rgb_map).any())

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    if DEBUG:
        print("white bkgd: "+str(white_bkgd))
        print(torch.isnan(weights_to_aggregate).any())
        print(torch.isnan(acc_map).any())
        print(torch.isnan(1.-acc_map[...,None]).any())
        print(torch.isnan(rgb_map + (1.-acc_map[...,None])).any())
        print(torch.isnan(rgb_map).any())
        print("===============")

    return rgb_map, disp_map, acc_map, weights, depth_map, tau, T

def sample_3sigma(low_3sigma, high_3sigma, N, det, near, far):
    t_vals = torch.linspace(0., 1., steps=N, device=device)
    step_size = (high_3sigma - low_3sigma) / (N - 1)
    bin_edges = (low_3sigma.unsqueeze(-1) * (1.-t_vals) + high_3sigma.unsqueeze(-1) * (t_vals)).clamp(near, far)
    factor = (bin_edges[..., 1:] - bin_edges[..., :-1]) / step_size.unsqueeze(-1)
    x_in_3sigma = torch.linspace(-3., 3., steps=(N - 1), device=device)
    bin_weights = factor * (1. / math.sqrt(2 * np.pi) * torch.exp(-0.5 * x_in_3sigma.pow(2))).unsqueeze(0).expand(*bin_edges.shape[:-1], N - 1)
    return sample_pdf(bin_edges, bin_weights, N, det=det)

def perturb_z_vals(z_vals, pytest):
    # get intervals between samples
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand_like(z_vals)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        t_rand = np.random.rand(*list(z_vals.shape))
        t_rand = torch.Tensor(t_rand)

    z_vals = lower + (upper - lower) * t_rand
    return z_vals

def compute_samples_around_depth(raw, z_vals, rays_d, N_samples, perturb, lower_bound, near, far, mode):
    sampling_depth, sampling_std = raw2depth(raw, near, far, z_vals, rays_d, mode)
    sampling_std = sampling_std.clamp(min=lower_bound)
    depth_min = sampling_depth - 3. * sampling_std
    depth_max = sampling_depth + 3. * sampling_std
    return sample_3sigma(depth_min, depth_max, N_samples, perturb == 0., near, far)

def forward_with_additonal_samples(z_vals, near, far, raw, z_vals_2, rays_o, rays_d, viewdirs, embedded_cam, network_fn, network_query_fn, raw_noise_std, pytest, mode, color_mode, farcolorfix=False):
    pts_2 = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_2[...,:,None]
    raw_2 = network_query_fn(pts_2, viewdirs, embedded_cam, network_fn)
    z_vals = torch.cat((z_vals, z_vals_2), -1)
    raw = torch.cat((raw, raw_2), 1)
    z_vals, indices = z_vals.sort()
    raw = torch.gather(raw, 1, indices.unsqueeze(-1).expand_as(raw))
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std, pytest=pytest, farcolorfix=farcolorfix)
    return {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights}

def render_rays(ray_batch,
                use_viewdirs,
                network_fn,
                network_query_fn,
                N_samples,
                mode,
                color_mode,
                precomputed_z_samples=None,
                embedded_cam=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                white_bkgd=False,                
                is_joint=False,
                cached_u= None,
                scale_sample_gradient=False,
                quad_solution_v2=False,
                zero_tol = 1e-4,
                epsilon = 1e-3,
                farcolorfix = False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = None
    depth_range = None
    if use_viewdirs:
        viewdirs = ray_batch[:,8:11]
        if ray_batch.shape[-1] > 11:
            depth_range = ray_batch[:,11:14]
    else:
        if ray_batch.shape[-1] > 8:
            depth_range = ray_batch[:,8:11]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)

    # sample and render rays for dense depth priors for nerf
    N_samples_half = N_samples // 2
    if precomputed_z_samples is not None:
        # compute a lower bound for the sampling standard deviation as the maximal distance between samples
        lower_bound = precomputed_z_samples[-1] - precomputed_z_samples[-2]
    # train time: use precomputed samples along the whole ray and additionally sample around the depth
    if depth_range is not None:
        valid_depth = depth_range[:,0] >= near[0, 0]
        invalid_depth = valid_depth.logical_not()
        # do a forward pass for the precomputed first half of samples
        z_vals = precomputed_z_samples.unsqueeze(0).expand((N_rays, N_samples_half))
        if perturb > 0.:
            z_vals = perturb_z_vals(z_vals, pytest)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
        z_vals_2 = torch.empty_like(z_vals)
        # sample around the predicted depth from the first half of samples, if the input depth is invalid
        z_vals_2[invalid_depth] = compute_samples_around_depth(raw.detach()[invalid_depth], z_vals[invalid_depth], rays_d[invalid_depth], N_samples_half, perturb, lower_bound, near[0, 0], far[0, 0], mode)
        # sample with in 3 sigma of the input depth, if it is valid
        z_vals_2[valid_depth] = sample_3sigma(depth_range[valid_depth, 1], depth_range[valid_depth, 2], N_samples_half, perturb == 0., near[0, 0], far[0, 0])
        return forward_with_additonal_samples(z_vals, raw, z_vals_2, rays_o, rays_d, viewdirs, embedded_cam, network_fn, network_query_fn, raw_noise_std, pytest, mode)
    # test time: use precomputed samples along the whole ray and additionally sample around the predicted depth from the first half of samples
    elif precomputed_z_samples is not None:
        z_vals = precomputed_z_samples.unsqueeze(0).expand((N_rays, N_samples_half))
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
        z_vals_2 = compute_samples_around_depth(raw, z_vals, rays_d, N_samples_half, perturb, lower_bound, near[0, 0], far[0, 0], mode)
        return forward_with_additonal_samples(z_vals, near, far, raw, z_vals_2, rays_o, rays_d, viewdirs, embedded_cam, network_fn, network_query_fn, raw_noise_std, pytest, mode)
    
    # sample and render rays for nerf
    elif not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb > 0.:
        z_vals = perturb_z_vals(z_vals, pytest)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd, farcolorfix=farcolorfix)

    if N_importance == 0:
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            if mode == "linear":
                z_samples, T_below, tau_below, bin_below, u = sample_pdf_reformulation_return_u(z_vals, weights, tau, T, near, far, N_samples, det=(perturb==0.), pytest=pytest, quad_solution_v2=quad_solution_v2)
            elif mode == "constant":
                z_samples, u = sample_pdf_return_u(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)
        else:
            if mode == "linear":
                z_samples, T_below, tau_below, bin_below, u = sample_pdf_reformulation_joint_return_u(z_vals, weights, tau, T, near, far, N_samples, det=(perturb==0.), pytest=pytest, quad_solution_v2=quad_solution_v2)
            elif mode == "constant":
                z_samples, u = sample_pdf_joint_return_u(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)

        if scale_sample_gradient:
            with torch.no_grad():
                pts = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None]
                samples_raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn)
            
            z_samples = Scale_Gradient_PDF.apply(z_samples, T_below, tau_below, bin_below, samples_raw)


        pred_depth_hyp = z_samples

    elif N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, z_vals_0, weights_0 = rgb_map, disp_map, acc_map, depth_map, z_vals, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        # z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)

        ### Hierarchical Sampling --> Fine NeRF
        
        if mode == "linear":
            z_samples, T_below, tau_below, bin_below = sample_pdf_reformulation(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest, quad_solution_v2=quad_solution_v2)
        elif mode == "constant":
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)

        z_samples = z_samples.detach()

        ### Debugging: There was some error on the intervals, left was not always < right
        z_samples = torch.clamp(z_samples, near, far)

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        if DEBUG:
            print("Does nan exist in pts")
            print(torch.isnan(pts).any())

        raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)
        
        if DEBUG:
            print("Does nan exist in forward")
            print(torch.isnan(raw).any())

        rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(raw, z_vals, near, far, rays_d, mode, color_mode, raw_noise_std, pytest=pytest, white_bkgd=white_bkgd, farcolorfix=farcolorfix)
        
        if DEBUG:
            print("Does nan/inf exist after converting to rgb outputs")
            print(torch.isnan(rgb_map).any())
            print(torch.isinf(rgb_map).any())

        ### P_depth from fine network
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            if mode == "linear":
                z_samples, T_below, tau_below, bin_below, u = sample_pdf_reformulation_return_u(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u, quad_solution_v2=quad_solution_v2)
            elif mode == "constant":
                z_samples, u = sample_pdf_return_u(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u)
        else:
            if mode == "linear":
                z_samples, T_below, tau_below, bin_below, u = sample_pdf_reformulation_joint_return_u(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u, quad_solution_v2=quad_solution_v2)
            elif mode == "constant":
                z_samples, u = sample_pdf_joint_return_u(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u)

        if scale_sample_gradient:
            ## Just get the magnitude, but it should be no_grad (I had a bug before? --> check this Mika)
            with torch.no_grad():
                pts = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None]
                samples_raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)
            
            z_samples = Scale_Gradient_PDF.apply(z_samples, T_below, tau_below, bin_below, samples_raw)

        pred_depth_hyp = z_samples


    if mode == "linear":
        weights = weights[..., 1:]

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights, 'pred_hyp' : pred_depth_hyp,\
    'u':u}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_vals0'] = z_vals_0
        ret['weights0'] = weights_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        # ret['pred_hyp'] = pred_depth_hyp

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def get_ray_batch_from_one_image(H, W, i_train, images, depths, valid_depths, poses, intrinsics, args):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    img_i = np.random.choice(i_train)
    target = images[img_i]
    target_depth = depths[img_i]
    target_valid_depth = valid_depths[img_i]
    pose = poses[img_i]
    intrinsic = intrinsics[img_i, :]
    rays_o, rays_d = get_rays(H, W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    select_coords = select_coordinates(coords, args.N_rand)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_d = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1) or (N_rand, 2)
    target_vd = target_valid_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
    if args.depth_loss_weight > 0.:
        depth_range = precompute_depth_sampling(target_d)
        batch_rays = torch.stack([rays_o, rays_d, depth_range], 0)  # (3, N_rand, 3)
    else:
        batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    return batch_rays, target_s, target_d, target_vd, img_i

def get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, intrinsics, all_hypothesis, args, space_carving_idx=None, cached_u=None, gt_valid_depths=None):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    # img_i = np.random.choice(i_train)
    
    target = images[img_i]
    target_depth = depths[img_i]
    target_valid_depth = valid_depths[img_i]
    pose = poses[img_i]
    intrinsic = intrinsics[img_i, :]

    target_hypothesis = all_hypothesis[img_i]

    rays_o, rays_d = get_rays(H, W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    select_coords = select_coordinates(coords, args.N_rand)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_d = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1) or (N_rand, 2)
    target_vd = target_valid_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
    target_h = target_hypothesis[:, select_coords[:, 0], select_coords[:, 1]]

    if space_carving_idx is not None:
        # print(space_carving_idx.shape)
        # print(space_carving_idx[img_i, select_coords[:, 0], select_coords[:, 1]].shape)
        target_hypothesis  = target_hypothesis.repeat(1, 1, 1, space_carving_idx.shape[-1])

        curr_space_carving_idx = space_carving_idx[img_i, select_coords[:, 0], select_coords[:, 1]]

        target_h_rays = target_hypothesis[ :, select_coords[:, 0], select_coords[:, 1]]

        target_h = torch.gather(target_h_rays, 1, curr_space_carving_idx.unsqueeze(0).long())


    if cached_u is not None:
        curr_cached_u = cached_u[img_i, select_coords[:, 0], select_coords[:, 1]]
    else:
        curr_cached_u = None

    # if args.mask_corners:
    #     ### Initialize a masked image
    #     space_carving_mask = torch.ones((target.shape[0], target.shape[1]), dtype=torch.float, device=images.device)

    #     ### Mask out the corners
    #     num_pix_to_mask = 20
    #     space_carving_mask[:num_pix_to_mask, :num_pix_to_mask] = 0
    #     space_carving_mask[:num_pix_to_mask, -num_pix_to_mask:] = 0
    #     space_carving_mask[-num_pix_to_mask:, :num_pix_to_mask] = 0
    #     space_carving_mask[-num_pix_to_mask:, -num_pix_to_mask:] = 0

    #     space_carving_mask = space_carving_mask[select_coords[:, 0], select_coords[:, 1]]
    # else:
    #     space_carving_mask = None

    space_carving_mask = gt_valid_depths[img_i].squeeze()
    space_carving_mask = space_carving_mask[select_coords[:, 0], select_coords[:, 1]]

    if args.depth_loss_weight > 0.:
        depth_range = precompute_depth_sampling(target_d)
        batch_rays = torch.stack([rays_o, rays_d, depth_range], 0)  # (3, N_rand, 3)
    else:
        batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    return batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u

def complete_depth(images, depths, valid_depths, input_h, input_w, model_path, invalidate_large_std_threshold=-1.):
    device = images.device

    # prepare input
    orig_size = (depths.shape[1], depths.shape[2])
    input_size = (input_h, input_w)
    images_tmp = images.permute(0, 3, 1, 2)
    depths_tmp = depths[..., 0]
    images_tmp = torchvision.transforms.functional.resize(images_tmp, input_size, \
        interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
    depths_tmp, valid_depths_tmp = resize_sparse_depth(depths_tmp, valid_depths, input_size)
    normalize, _ = get_pretrained_normalize()
    depths_tmp[valid_depths_tmp] = convert_m_to_depth_completion_scaling(depths_tmp[valid_depths_tmp])

    # run depth completion
    with torch.no_grad():
        net = resnet18_skip(pretrained=False, map_location=device, input_size=input_size).to(device)
        net.eval()
        ckpt = torch.load(model_path)
        missing_keys, unexpected_keys = net.load_state_dict(ckpt['network_state_dict'], strict=False)
        print("Loading model: \n  missing keys: {}\n  unexpected keys: {}".format(missing_keys, unexpected_keys))
        
        depths_out = torch.empty_like(depths_tmp)
        depths_std_out = torch.empty_like(depths_tmp)
        for i, (rgb, depth) in enumerate(zip(images_tmp, depths_tmp)):
            rgb = normalize['rgb'](rgb)
            input = torch.cat((rgb, depth.unsqueeze(0)), 0).unsqueeze(0)
            pred = net(input)
            depths_out[i] = convert_depth_completion_scaling_to_m(pred[0])
            depths_std_out[i] = convert_depth_completion_scaling_to_m(pred[1])
        depths_out = torch.stack((depths_out, depths_std_out), 1)
        depths_out = torchvision.transforms.functional.resize(depths_out, orig_size, \
            interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
    
    # apply max min filter
    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1)
    depths_out_0 = depths_out.narrow(1, 0, 1).clamp(min=0)
    depths_out_max = max_pool(depths_out_0) + 0.01
    depths_out_min = -1. * max_pool(-1. * depths_out_0) - 0.01
    depths_out[:, 1, :, :] = torch.maximum(depths_out[:, 1, :, :], (depths_out_max - depths_out_min).squeeze(1))
    
    # mask out depth with very large uncertainty
    depths_out = depths_out.permute(0, 2, 3, 1)
    valid_depths_out = torch.full_like(valid_depths, True)
    if invalidate_large_std_threshold > 0.:
        large_std_mask = depths_out[:, :, :, 1] > invalidate_large_std_threshold
        valid_depths_out[large_std_mask] = False
        depths_out[large_std_mask] = 0.
        print("Masked out {:.1f} percent of completed depth with standard deviation greater {:.2f}".format( \
            100. * (1. - valid_depths_out.sum() / valid_depths_out.numel()), invalidate_large_std_threshold))
    
    return depths_out, valid_depths_out

def complete_and_check_depth(images, depths, valid_depths, i_train, gt_depths_train, gt_valid_depths_train, scene_sample_params, args):
    near, far = scene_sample_params["near"], scene_sample_params["far"]

    # print statistics before completion
    eval_mask = torch.logical_and(gt_valid_depths_train, valid_depths[i_train])
    p_complete = valid_depths[i_train].sum() * 100. / valid_depths[i_train].numel()
    rmse_before = compute_rmse(depths[i_train, :, :, 0][eval_mask], gt_depths_train.squeeze(-1)[eval_mask])
    print("Depth maps are {:.4f} percent complete and have RMSE {:.4f} before completion".format(p_complete, rmse_before))

    # add channel for depth standard deviation and run depth completion
    depths_std = torch.zeros_like(depths)
    depths = torch.cat((depths, depths_std), 3)
    depths[i_train], valid_depths[i_train] = complete_depth(images[i_train], depths[i_train], valid_depths[i_train], \
        args.depth_completion_input_h, args.depth_completion_input_w, args.depth_prior_network_path, \
        invalidate_large_std_threshold=args.invalidate_large_std_threshold)

    # print statistics after completion
    depths[:, :, :, 0][valid_depths] = depths[:, :, :, 0][valid_depths].clamp(min=near, max=far)
    print("Completed depth maps in range {:.4f} - {:.4f}".format(depths[i_train, :, :, 0][valid_depths[i_train]].min(), \
        depths[i_train, :, :, 0][valid_depths[i_train]].max()))
    eval_mask = torch.logical_and(gt_valid_depths_train, valid_depths[i_train])
    print("Depth maps have RMSE {:.4f} after completion".format(compute_rmse(depths[i_train, :, :, 0][eval_mask], \
        gt_depths_train.squeeze(-1)[eval_mask])))
    lower_bound = 0.03
    depths[:, :, :, 1][valid_depths] = depths[:, :, :, 1][valid_depths].clamp(min=lower_bound, max=(far - near))
    print("Depth standard deviations in range {:.4f} - {:.4f}, with mean {:.4f}".format(depths[i_train, :, :, 1][valid_depths[i_train]].min(), \
        depths[i_train, :, :, 1][valid_depths[i_train]].max(), depths[i_train, :, :, 1][valid_depths[i_train]].mean()))

    return depths, valid_depths

def train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis, is_init_scales=False, scales_init=None, shifts_init=None):
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    tb = SummaryWriter(log_dir=os.path.join("runs", args.ckpt_dir, args.expname))
    near, far = scene_sample_params['near'], scene_sample_params['far']
    H, W = images.shape[1:3]
    i_train, i_val, i_test, i_video = i_split
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)
    print('TEST views are', i_test)

    # use ground truth depth for validation and test if available
    if gt_depths is not None:
        depths[i_test] = gt_depths[i_test]
        valid_depths[i_test] = gt_valid_depths[i_test]
        depths[i_val] = gt_depths[i_val]
        valid_depths[i_val] = gt_valid_depths[i_val]
    i_relevant_for_training = np.concatenate((i_train, i_val), 0)
    if len(i_test) == 0:
        print("Error: There is no test set")
        exit()
    if len(i_val) == 0:
        print("Warning: There is no validation set, test set is used instead")
        i_val = i_test
        i_relevant_for_training = np.concatenate((i_relevant_for_training, i_val), 0)

    # keep test data on cpu until needed
    test_images = images[i_test]

    if depths is not None:
        test_depths = depths[i_test]
        test_valid_depths = valid_depths[i_test]
    else:
        test_depths = torch.zeros((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)).to(device)
        test_valid_depths = torch.zeros((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)).to(device)        
    
    test_poses = poses[i_test]
    test_intrinsics = intrinsics[i_test]
    i_test = i_test - i_test[0]

    # move training data to gpu
    images = torch.Tensor(images[i_relevant_for_training]).to(device)
    
    if depths is not None:
        depths = torch.Tensor(depths[i_relevant_for_training]).to(device)
        valid_depths = torch.Tensor(valid_depths[i_relevant_for_training]).bool().to(device)
    else:
        depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], 1)).to(device)
        valid_depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], 1)).to(device)

    poses = torch.Tensor(poses[i_relevant_for_training]).to(device)
    intrinsics = torch.Tensor(intrinsics[i_relevant_for_training]).to(device)
    all_depth_hypothesis = torch.Tensor(all_depth_hypothesis).to(device)

    if gt_depths is not None:
        # complete and check depth
        gt_depths_train = torch.Tensor(gt_depths[i_train]).to(device) # only used to evaluate error of completed depth
        gt_valid_depths_train = torch.Tensor(gt_valid_depths[i_train]).bool().to(device) # only used to evaluate error of completed depth


    ###########################################################################
    #### Use GT depth for space carving --> overriding all_depth_hypothesis ###
    gt_depths_train = gt_depths_train.unsqueeze(1)
    gt_valid_depths_train = gt_valid_depths_train.unsqueeze(1)
    ###########################################################################

    ##### Initialize depth scale and shift
    DEPTH_SCALES = torch.autograd.Variable(torch.ones((images.shape[0], 1), dtype=torch.float, device=images.device)*args.scale_init, requires_grad=True)
    DEPTH_SHIFTS = torch.autograd.Variable(torch.ones((images.shape[0], 1), dtype=torch.float, device=images.device)*args.shift_init, requires_grad=True)      

    print(DEPTH_SCALES)
    print()
    print(DEPTH_SHIFTS)
    print()
    print(DEPTH_SCALES.shape)
    print(DEPTH_SHIFTS.shape)

    optimizer_ss = torch.optim.Adam(params=(DEPTH_SCALES, DEPTH_SHIFTS,), lr=args.scaleshift_lr)
    
    print("Done with scale and shift init.")
    ################################

    # create nerf model
    render_kwargs_train, render_kwargs_test, start, nerf_grad_vars, optimizer = create_nerf(args, scene_sample_params)
    
    # create camera embedding function
    embedcam_fn = None
    if args.input_ch_cam > 0:
        embedcam_fn = torch.nn.Embedding(len(i_train), args.input_ch_cam)

    # optimize nerf
    print('Begin')
    N_iters = args.num_iterations + 1
    global_step = start
    start = start + 1

    #### For cIMLE ###
    # if args.cimle_white_balancing and args.load_pretrained:
    if args.load_pretrained:
        ########################################
        #### Load pretrained model w/o cIMLE ###
        ########################################
        path = args.pretrained_dir
        ckpts = [os.path.join(path, f) for f in sorted(os.listdir(path)) if '000.tar' in f]
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reloading pretrained model from', ckpt_path)

        ckpt = torch.load(ckpt_path)

        coarse_model_dict = render_kwargs_train["network_fn"].state_dict()
        coarse_keys = {k: v for k, v in ckpt['network_fn_state_dict'].items() if k in coarse_model_dict} 

        fine_model_dict = render_kwargs_train["network_fine"].state_dict()
        fine_keys = {k: v for k, v in ckpt['network_fine_state_dict'].items() if k in fine_model_dict} 

        print(len(coarse_keys.keys()))
        print(len(fine_keys.keys()))

        ### Load weights from pretrained model without cIMLE
        print("Num keys loaded:")
        coarse_model_dict.update(coarse_keys)
        fine_model_dict.update(fine_keys)

        ## Load scale and shift
        DEPTH_SHIFTS = torch.load(ckpt_path)["depth_shifts"]
        DEPTH_SCALES = torch.load(ckpt_path)["depth_scales"] 

        print("Scales:")
        print(DEPTH_SCALES)
        print()
        print("Shifts:")
        print(DEPTH_SHIFTS)

        print("Loaded depth shift/scale from pretrained model.")
        ########################################
        ########################################        

    if args.cimle_white_balancing:
        NUM_SAMPLES = args.num_samples_whitebalancing
        print("Num samples for cIMLE:")
        print(NUM_SAMPLES)

     
    SPACE_CARVING_INDICES = None
    CACHED_U = None
    ##################

    for i in trange(start, N_iters):

        # update learning rate
        if i > args.start_decay_lrate and i <= args.end_decay_lrate:
            portion = (i - args.start_decay_lrate) / (args.end_decay_lrate - args.start_decay_lrate)
            decay_rate = 0.1
            new_lrate = args.lrate * (decay_rate ** portion)
            update_learning_rate(optimizer, new_lrate)

        ##### For  cIMLE ###
        if args.cimle_white_balancing:
            ###############################################
            ########## cIMLE on the latent code ###########
            ###############################################
            if (args.input_ch_cam > 0) and (i % args.refresh_z_whitebalancing == 3 or i == 1):
                print("Recaching z-codes.")

                ### Recacahe per image latent codes
                num_images = len(i_train)
                all_losses = torch.zeros((NUM_SAMPLES, num_images), device=images.device)
                all_z = torch.normal(0.0, 1.0, size=(NUM_SAMPLES, num_images, args.input_ch_cam), device=images.device)

                ### Sample NUM_SAMPLE times
                for s in range(NUM_SAMPLES):
                    ### Loop through the images
                    for n, img_idx in enumerate(i_train):
                        print("Recaching image {}/{}".format(n + 1, num_images), end="")
                        target = images[img_idx]
                        pose = poses[img_idx, :3,:4]
                        intrinsic = intrinsics[img_idx, :]

                        render_kwargs_test["embedded_cam"] = all_z[s, img_idx]

                        with torch.no_grad():
                            rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 4), c2w=pose, **render_kwargs_test)
                        
                            # compute color metrics
                            img_loss = img2mse(rgb, target)
                            print(img_loss)
                            
                            all_losses[s, img_idx] = img_loss


                ### Get the best loss and select and z code
                idx_to_take = torch.argmin(all_losses, axis=0)

                selected_z_np = np.empty((num_images, args.input_ch_cam), dtype=np.float32)
                for n, img_idx in enumerate(i_train):
                    selected_z_np = all_z[idx_to_take[i_train]][img_idx].cpu().data.numpy()

                selected_z_np = torch.from_numpy(selected_z_np).to(images.device)
                embedcam_fn = selected_z_np
            ###############################################

        ### Only start cIMLE after certain number of iterations
        if args.cimle_nerf and i % args.refresh_z_nerf == 12 and i>=args.start_z_nerf:
        # if args.cimle_nerf and (i % args.refresh_z_nerf == 701 or i == 1):
            ###############################################
            #### cIMLE on NeRF for Space Carving ##########
            ###############################################
            with torch.no_grad():

                torch.cuda.empty_cache()               
                num_images = len(i_train)
                SPACE_CARVING_INDICES = torch.zeros((num_images, images.shape[1], images.shape[2], args.N_importance), dtype=torch.int, device=images.device)
                CACHED_U = torch.zeros((num_images, images.shape[1], images.shape[2], args.N_importance), dtype=torch.float, device=images.device)

                print("Re-caching indices for space carving loss.")
                for n, img_idx in enumerate(i_train):
                    print("Recaching image {}/{}".format(n + 1, num_images), end="")
                    target = images[img_idx]
                    pose = poses[img_idx, :3,:4]
                    intrinsic = intrinsics[img_idx, :]
                    prior_depth_hypothesis = gt_depths_train[img_idx]

                    ### Rescale with current scale shift values
                    curr_scale = DEPTH_SCALES[img_idx]
                    curr_shift = DEPTH_SHIFTS[img_idx]
                    prior_depth_hypothesis = prior_depth_hypothesis*curr_scale + curr_shift                 

                    if args.input_ch_cam > 0:
                        render_kwargs_test['embedded_cam'] = embedcam_fn[img_idx]

                    rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 2), c2w=pose, **render_kwargs_test)
                    # rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 3), c2w=pose, **render_kwargs_test)

                    u = extras['u']

                    curr_space_carving_idx = get_space_carving_idx_corrected(extras["pred_hyp"], prior_depth_hypothesis, is_joint=args.is_joint, norm_p=args.norm_p, threshold=args.space_carving_threshold)
                    
                    SPACE_CARVING_INDICES[img_idx] = curr_space_carving_idx
                    CACHED_U[img_idx] = u


                ###############################################
                # print(SPACE_CARVING_INDICES)
                # print()
        ####################
        
        ### Scale the hypotheses by scale and shift
        img_i = np.random.choice(i_train)

        curr_scale = DEPTH_SCALES[img_i]
        curr_shift = DEPTH_SHIFTS[img_i]

        ## Scale and shift
        batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u = get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, \
            intrinsics, gt_depths_train, args, SPACE_CARVING_INDICES, CACHED_U, gt_valid_depths_train)

        target_h = target_h*curr_scale + curr_shift        

        if args.input_ch_cam > 0:
            render_kwargs_train['embedded_cam'] = embedcam_fn[img_i]

        target_d = target_d.squeeze(-1)

        # render
        ## Feed cached quantiles into the renderer
        render_kwargs_train["cached_u"] = curr_cached_u

        rgb, _, _, extras = render_hyp(H, W, None, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,  is_joint=args.is_joint, \
            quad_solution_v2=args.quad_solution_v2, scale_sample_gradient = args.scale_sample_gradient, **render_kwargs_train)

        # compute loss and optimize
        optimizer.zero_grad()
        optimizer_ss.zero_grad()
        img_loss = img2mse(rgb, target_s)
        psnr = mse2psnr(img_loss)
        
        loss = img_loss

        if args.space_carving_weight>0. and i>args.warm_start_nerf:
            
            # space_carving_loss = compute_space_carving_loss(extras["pred_hyp"], target_h, is_joint=args.is_joint, norm_p=args.norm_p, threshold=args.space_carving_threshold, mask=space_carving_mask)
            space_carving_loss = compute_space_carving_loss_corrected(extras["pred_hyp"], target_h, is_joint=args.is_joint, norm_p=args.norm_p, threshold=args.space_carving_threshold, mask=space_carving_mask)
            
            loss = loss + args.space_carving_weight * space_carving_loss
        else:
            space_carving_loss = torch.mean(torch.zeros([target_h.shape[0]]).to(target_h.device))

        if args.depth_loss_weight > 0.:
            depth_loss = compute_depth_loss(extras['depth_map'], extras['z_vals'], extras['weights'], target_d, target_vd)
            loss = loss + args.depth_loss_weight * depth_loss
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            psnr0 = mse2psnr(img_loss0)
            loss = loss + img_loss0

        loss.backward()
        nn.utils.clip_grad_value_(nerf_grad_vars, 0.1)
        optimizer.step()

        ### Don't optimize scale shift for the last 100k epochs, check whether the appearance will crisp
        if i < args.freeze_ss:
            optimizer_ss.step()

        ### Update camera embeddings
        if args.input_ch_cam > 0 and args.opt_ch_cam:
            optimizer_latent.step() 

        # write logs
        if i%args.i_weights==0:
            path = os.path.join(args.ckpt_dir, args.expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()

            if args.input_ch_cam > 0:
                save_dict['embedded_cam'] = embedcam_fn

            save_dict['depth_shifts'] = DEPTH_SHIFTS
            save_dict['depth_scales'] = DEPTH_SCALES

            torch.save(save_dict, path)
            print('Saved checkpoints at', path)
        
        if i%args.i_print==0:
            tb.add_scalars('mse', {'train': img_loss.item()}, i)

            if args.space_carving_weight > 0.:
                tb.add_scalars('space_carving_loss', {'train': space_carving_loss.item()}, i)

            if args.depth_loss_weight > 0.:
                tb.add_scalars('depth_loss', {'train': depth_loss.item()}, i)

            tb.add_scalars('psnr', {'train': psnr.item()}, i)
            if 'rgb0' in extras:
                tb.add_scalars('mse0', {'train': img_loss0.item()}, i)
                tb.add_scalars('psnr0', {'train': psnr0.item()}, i)

            scale_mean = torch.mean(DEPTH_SCALES[i_train])
            shift_mean = torch.mean(DEPTH_SHIFTS[i_train])
            tb.add_scalars('depth_scale_mean', {'train': scale_mean.item()}, i)
            tb.add_scalars('depth_shift_mean', {'train': shift_mean.item()}, i) 

            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  MSE: {img_loss.item()} Space carving: {space_carving_loss.item()}")
            
        if i%args.i_img==0:
            # visualize 2 train images
            _, images_train = render_images_with_metrics(2, i_train, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, embedcam_fn=embedcam_fn)
            tb.add_image('train_image',  torch.cat((
                torchvision.utils.make_grid(images_train["rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_train["depths"], nrow=1), \
                torchvision.utils.make_grid(images_train["target_depths"], nrow=1)), 2), i)
            # compute validation metrics and visualize 8 validation images
            mean_metrics_val, images_val = render_images_with_metrics(8, i_val, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
            tb.add_scalars('mse', {'val': mean_metrics_val.get("img_loss")}, i)
            tb.add_scalars('psnr', {'val': mean_metrics_val.get("psnr")}, i)
            tb.add_scalar('ssim', mean_metrics_val.get("ssim"), i)
            tb.add_scalar('lpips', mean_metrics_val.get("lpips"), i)
            if mean_metrics_val.has("depth_rmse"):
                tb.add_scalar('depth_rmse', mean_metrics_val.get("depth_rmse"), i)
            if 'rgbs0' in images_val:
                tb.add_scalars('mse0', {'val': mean_metrics_val.get("img_loss0")}, i)
                tb.add_scalars('psnr0', {'val': mean_metrics_val.get("psnr0")}, i)
            if 'rgbs0' in images_val:
                tb.add_image('val_image',  torch.cat((
                    torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["rgbs0"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                    torchvision.utils.make_grid(images_val["depths0"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2), i)
            else:
                tb.add_image('val_image',  torch.cat((
                    torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                    torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                    torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2), i)

        # test at the last iteration
        if (i + 1) == N_iters:
            torch.cuda.empty_cache()
            images = torch.Tensor(test_images).to(device)
            depths = torch.Tensor(test_depths).to(device)
            valid_depths = torch.Tensor(test_valid_depths).bool().to(device)
            poses = torch.Tensor(test_poses).to(device)
            intrinsics = torch.Tensor(test_intrinsics).to(device)
            mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
            write_images_with_metrics(images_test, mean_metrics_test, far, args)
            tb.flush()

        global_step += 1


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('task', type=str, help='one out of: "train", "test", "test_with_opt", "video"')
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default=None, 
                        help='specify the experiment, required for "test" and "video", optional for "train"')
    parser.add_argument("--dataset", type=str, default="scannet", 
                        help='dataset used -- selects which dataloader"')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32,
                        help='batch size (number of random rays per gradient step)')


    parser.add_argument('--num_iterations', type=int, default=500000, help='Number of epochs')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--start_decay_lrate", type=int, default=400000, 
                        help='start iteration for learning rate decay')
    parser.add_argument("--end_decay_lrate", type=int, default=500000, 
                        help='end iteration for learning rate decay')


    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64*4, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--depth_loss_weight", type=float, default=0.004,
                        help='weight of the depth loss, values <=0 do not apply depth loss')
    parser.add_argument("--invalidate_large_std_threshold", type=float, default=1.,
                        help='invalidate completed depth values with standard deviation greater than threshold in m, \
                            thresholds <=0 deactivate invalidation')
    parser.add_argument("--depth_completion_input_h", type=int, default=240, 
                        help='depth completion network input height')
    parser.add_argument("--depth_completion_input_w", type=int, default=320, 
                        help='depth completion network input width')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=256,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=9,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=0,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--lindisp", action='store_true', default=False,
                        help='sampling linearly in disparity rather than depth')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=20000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help='checkpoint directory')

    # data options
    parser.add_argument("--scene_id", type=str, default="scene0710_00",
                        help='scene identifier')
    parser.add_argument("--depth_prior_network_path", type=str, default="",
                        help='path to the depth prior network checkpoint to be used')
    parser.add_argument("--data_dir", type=str, default="",
                        help='directory containing the scenes')

    ### Train json file --> experimenting making views sparser
    parser.add_argument("--train_jsonfile", type=str, default='transforms_train.json',
                        help='json file containing training images')

    ### For Blender dataset ###
    #### training options --> not implemented yet
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')    
    #########

    parser.add_argument("--random_seed",   type=int, default=0, 
                        help='random seed used')


    ##################
    ### For SCADE ###
    ##################
    parser.add_argument("--cimle_dir", type=str, default="dump_0826_pretrained_dd_scene0710_train/",
                        help='dump_dir name for prior depth hypotheses')
    parser.add_argument("--num_hypothesis", type=int, default=20, 
                        help='number of cimle hypothesis')
    parser.add_argument("--space_carving_weight", type=float, default=0.007,
                        help='weight of the depth loss, values <=0 do not apply depth loss')
    parser.add_argument("--warm_start_nerf", type=int, default=0, 
                        help='number of iterations to train only vanilla nerf without additional losses.')

    parser.add_argument('--scaleshift_lr', default= 0.000001, type=float)
    parser.add_argument('--scale_init', default= 1.0, type=float)
    parser.add_argument('--shift_init', default= 0.0, type=float)
    parser.add_argument("--freeze_ss", type=int, default=0, 
                            help='dont update scale/shift starting epoch X')

    ### u sampling is joint or not
    parser.add_argument('--is_joint', default= False, type=bool)

    ### Norm for space carving loss
    parser.add_argument("--norm_p", type=int, default=2, help='norm for loss')
    parser.add_argument("--space_carving_threshold", type=float, default=0.0,
                        help='threshold to not penalize the space carving loss.')
    parser.add_argument('--mask_corners', default= False, type=bool)

    parser.add_argument('--load_pretrained', default= False, type=bool)
    parser.add_argument("--pretrained_dir", type=str, default="Scannet/scene758/scene0758_00_sc007_sslr1e-07",
                        help='folder directory name for where the pretrained model that we want to load is')


    ### For cIMLE on nerf-prior training
    parser.add_argument('--cimle_nerf', default= False, type=bool)
    parser.add_argument('--refresh_z_nerf', default= 50000, type=int, help='Number of iterations to recache latent code')
    parser.add_argument('--start_z_nerf', default= 100000, type=int, help='Number of iterations to recache latent code')


    ###### For cIMLE for white balancing
    parser.add_argument('--cimle_white_balancing', default= False, type=bool)


    parser.add_argument("--cimle_lrate", type=float, default=5e-5, 
                        help='learning rate')    
    parser.add_argument('--refresh_z_whitebalancing', default= 50000, type=int, help='Number of iterations to recache latent code')
    parser.add_argument('--num_samples_whitebalancing', default= 20, type=int, help='Number samples drawn at each recache')

    parser.add_argument("--input_ch_cam", type=int, default=0,
                        help='number of channels for camera index embedding')

    parser.add_argument("--opt_ch_cam", action='store_true', default=False,
                        help='optimize camera embedding')    
    parser.add_argument('--ch_cam_lr', default= 0.0001, type=float)

    ##################
    ##################


    ### For PWL formulation
    parser.add_argument("--mode", type=str, default="constant", 
                        help='rendering aggregation mode -- whether to use piecewise constant (vanilla) or piecewise linear (reformulation)."')
    parser.add_argument("--color_mode", type=str, default="midpoint", 
                        help='rendering color aggregation mode -- whether to use left bin or midpoint."')

    parser.add_argument('--scale_sample_gradient', default= False, type=bool)
    parser.add_argument('--quad_solution_v2', default= True, type=bool)
    parser.add_argument("--zero_tol", type=float, default=1e-4, 
                        help='zero tol to revert to piecewise constant assumption')    
    parser.add_argument("--epsilon", type=float, default=1e-3, 
                        help='epsilon value in the increasing and decreasing cases or max(x,epsilon)')

    parser.add_argument('--set_near_plane', default= 0.5, type=float)
    parser.add_argument('--farcolorfix', default= False, type=bool)

    parser.add_argument('--train_skip', default= 1, type=int, help='Number training images to skip for sparsity')

    return parser

def run_nerf():
    
    parser = config_parser()
    args = parser.parse_args()

    if args.task == "train":
        if args.expname is None:
            args.expname = "{}_{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'), args.scene_id)
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        os.makedirs(os.path.join(args.ckpt_dir, args.expname), exist_ok=True)
        with open(args_file, 'w') as af:
            json.dump(vars(args), af, indent=4)

    elif args.task == "test_samples":
        if args.expname is None:
            print("Error: Specify experiment name for test or video")
            exit()
        tmp_task = args.task
        tmp_data_dir = args.data_dir
        tmp_ckpt_dir = args.ckpt_dir
        tmp_N_samples = args.N_samples
        tmp_N_importance = args.N_importance
        tmp_mode = args.mode
        tmp_set_near_plane = args.set_near_plane

        # load nerf parameters from training
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        with open(args_file, 'r') as af:
            args_dict = json.load(af)
        args_loaded = Namespace(**args_dict)
        # task and paths are not overwritten
        args.task = tmp_task
        args.data_dir = tmp_data_dir
        args.ckpt_dir = tmp_ckpt_dir
        args.train_jsonfile = 'transforms_train.json'

        args.N_samples = tmp_N_samples
        args.N_importance = tmp_N_importance
        args.mode = tmp_mode
        args.set_near_plane = tmp_set_near_plane

    else:
        if args.expname is None:
            print("Error: Specify experiment name for test or video")
            exit()
        tmp_task = args.task
        tmp_data_dir = args.data_dir
        tmp_ckpt_dir = args.ckpt_dir
        tmp_set_near_plane = args.set_near_plane
        tmp_mode = args.mode
        tmp_N_samples = args.N_samples
        tmp_N_importance = args.N_importance

        # load nerf parameters from training
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
        with open(args_file, 'r') as af:
            args_dict = json.load(af)
        args = Namespace(**args_dict)
        # task and paths are not overwritten
        args.task = tmp_task
        args.data_dir = tmp_data_dir
        args.ckpt_dir = tmp_ckpt_dir
        args.mode = tmp_mode
        args.train_jsonfile = 'transforms_train.json'
        args.set_near_plane = tmp_set_near_plane
        args.N_samples = tmp_N_samples
        args.N_importance = tmp_N_importance

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    scene_data_dir = os.path.join(args.data_dir, args.scene_id)

    if args.dataset == "scannet":
        images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
        gt_depths, gt_valid_depths, all_depth_hypothesis = load_scene_mika(scene_data_dir, args.cimle_dir, args.num_hypothesis, 'transforms_train.json')

    elif args.dataset == "blender_depth":
        images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths = load_scene_blender_depth(scene_data_dir, half_res=args.half_res, train_skip=args.train_skip, near_plane=args.set_near_plane)

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]  

    elif args.dataset == "blender2_depth":
        images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths = load_scene_blender2_depth(scene_data_dir, half_res=args.half_res, train_skip=args.train_skip, near_plane=args.set_near_plane)

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]  
    else:
        print("ERROR: Dataloader not implemented for dataset: "+args.dataset)
        exit()

    if args.dataset == "blender_depth" or args.dataset == "blender2_depth":
        near = args.set_near_plane
        print("Set near plane to: " + str(near))

        ### Make the depth hypothesis same as the laoded depth
        all_depth_hypothesis = depths[:, np.newaxis, :, :]

    i_train, i_val, i_test, i_video = i_split


    # Compute boundaries of 3D space
    max_xyz = torch.full((3,), -1e6, device=device)
    min_xyz = torch.full((3,), 1e6, device=device)
    for idx_train in i_train:
        rays_o, rays_d = get_rays(H, W, torch.Tensor(intrinsics[idx_train]).to(device), torch.Tensor(poses[idx_train]).to(device)) # (H, W, 3), (H, W, 3)
        points_3D = rays_o + rays_d * far # [H, W, 3]
        max_xyz = torch.max(points_3D.view(-1, 3).amax(0), max_xyz)
        min_xyz = torch.min(points_3D.view(-1, 3).amin(0), min_xyz)
    args.bb_center = (max_xyz + min_xyz) / 2.
    # args.bb_scale = 2. / (max_xyz - min_xyz).max()

    ### Make scale 1.
    # args.bb_center = 0.0
    args.bb_scale = 1.0
    print("Computed scene boundaries: min {}, max {}".format(min_xyz, max_xyz))

    # Precompute scene sampling parameters
    if args.depth_loss_weight > 0.:
        precomputed_z_samples = precompute_quadratic_samples(near, far, args.N_samples // 2)

        if precomputed_z_samples.shape[0] % 2 == 1:
            precomputed_z_samples = precomputed_z_samples[:-1]

        print("Computed {} samples between {} and {}".format(precomputed_z_samples.shape[0], precomputed_z_samples[0], precomputed_z_samples[-1]))
    else:
        precomputed_z_samples = None
    scene_sample_params = {
        'precomputed_z_samples' : precomputed_z_samples,
        'near' : near,
        'far' : far,
    }

    lpips_alex = LPIPS()

    if args.task == "train":
        train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis)
        exit()

    # create nerf model for testing
    _, render_kwargs_test, _, nerf_grad_vars, _ = create_nerf(args, scene_sample_params)
    for param in nerf_grad_vars:
        param.requires_grad = False

    # render test set and compute statistics
    if "test" in args.task: 
        with_test_time_optimization = False
        if args.task == "test_opt":
            with_test_time_optimization = True
        images = torch.Tensor(images[i_test]).to(device)
        if gt_depths is None:
            depths = torch.Tensor(depths[i_test]).to(device)
            valid_depths = torch.Tensor(valid_depths[i_test]).bool().to(device)
        else:
            depths = torch.Tensor(gt_depths[i_test]).to(device)
            valid_depths = torch.Tensor(gt_valid_depths[i_test]).bool().to(device)
        poses = torch.Tensor(poses[i_test]).to(device)
        intrinsics = torch.Tensor(intrinsics[i_test]).to(device)
        i_test = i_test - i_test[0]
        mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=with_test_time_optimization)

        if "samples" in args.task:
            write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=with_test_time_optimization, test_samples=True)
        else:
            write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=with_test_time_optimization)
    elif args.task == "video":
        vposes = torch.Tensor(poses[i_video]).to(device)
        vintrinsics = torch.Tensor(intrinsics[i_video]).to(device)
        render_video(vposes, H, W, vintrinsics, str(0), args, render_kwargs_test)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    run_nerf()
