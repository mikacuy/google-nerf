'''
Mikaela Uy
mikacuy@stanford.edu
For Scannet data
Modified from DDP codebase
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

from model import NeRF_semantics, MotionPotential, get_embedder, get_rays, sample_pdf, sample_pdf_joint, img2mse, mse2psnr, to8b, \
    compute_depth_loss, select_coordinates, to16b, compute_space_carving_loss, \
    sample_pdf_return_u, sample_pdf_joint_return_u
from data import create_random_subsets, load_llff_data_multicam_withdepth, convert_depth_completion_scaling_to_m, \
    convert_m_to_depth_completion_scaling, get_pretrained_normalize, resize_sparse_depth, load_scene_mika, load_scene_blender_depth, load_scene_blender_depth_features, read_feature
from train_utils import MeanTracker, update_learning_rate, get_learning_rate
from metric import compute_rmse

import imageio
from natsort import natsorted 

from sklearn.decomposition import PCA
import PIL.Image
from PIL import Image

from utils_viz import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

IS_MOTION_DEBUG = True
# IS_MOTION_DEBUG = False

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
    inputs_flat = (inputs_flat - bb_center) * bb_scale
    embedded = embed_fn(inputs_flat) # samples * rays, multires * 2 * 3 + 3

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs, embedded_cam.unsqueeze(0).expand(embedded_dirs.shape[0], embedded_cam.shape[0])], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def run_motion_potential(inputs, features, fn, bb_center, bb_scale, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    inputs_flat = (inputs_flat - bb_center) * bb_scale
    embedded = torch.cat([inputs_flat, features], -1)

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


def render_video(poses, H, W, intrinsics, filename, args, render_kwargs_test, fps=10, features=None):
    video_dir = os.path.join(args.ckpt_dir, args.expname, 'video_' + filename)
    video_depth_dir = os.path.join(args.ckpt_dir, args.expname, 'video_depth_' + filename)
    video_depth_colored_dir = os.path.join(args.ckpt_dir, args.expname, 'video_depth_colored_' + filename)
    video_feat_dir = os.path.join(args.ckpt_dir, args.expname, 'video_feat_' + filename)

    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    if os.path.exists(video_depth_dir):
        shutil.rmtree(video_depth_dir)
    if os.path.exists(video_depth_colored_dir):
        shutil.rmtree(video_depth_colored_dir)        
    if os.path.exists(video_feat_dir):
        shutil.rmtree(video_feat_dir)   
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(video_depth_dir, exist_ok=True)
    os.makedirs(video_depth_colored_dir, exist_ok=True)
    os.makedirs(video_feat_dir, exist_ok=True)

    depth_scale = render_kwargs_test["far"]
    max_depth_in_video = 0

    idx_to_take = range(0, len(poses), 3)
    # idx_to_take = range(0, len(poses), 10)
    pred_feats_res = torch.empty(len(idx_to_take), H, W, args.feat_dim)

    for n in range(len(idx_to_take)):
    # for img_idx in range(200):
        img_idx = idx_to_take[n]
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]
        with torch.no_grad():
            if args.input_ch_cam > 0:
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
            # render video in 16:9 with one third rgb, one third depth and one third depth standard deviation
            rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 8), c2w=pose, with_5_9=False, **render_kwargs_test)
            rgb_cpu_numpy_8b = to8b(rgb.cpu().numpy())
            video_frame = cv2.cvtColor(rgb_cpu_numpy_8b, cv2.COLOR_RGB2BGR)

            pred_features = extras["feature_map"]
            pred_feats_res[n] = pred_features.cpu()

            max_depth_in_video = max(max_depth_in_video, extras['depth_map'].max())
            depth_colored_frame = cv2.applyColorMap(to8b((extras['depth_map'] / depth_scale).cpu().numpy()), cv2.COLORMAP_TURBO)
            depth = (extras['depth_map']).cpu().numpy()*1000.
            depth = (depth).astype(np.uint16)

            # max_depth_in_video = max(max_depth_in_video, extras['depth_map'].max())
            # depth_frame = cv2.applyColorMap(to8b((extras['depth_map'] / depth_scale).cpu().numpy()), cv2.COLORMAP_TURBO)
            # video_frame = np.concatenate((video_frame, depth_frame), 1)
            # depth_var = ((extras['z_vals'] - extras['depth_map'].unsqueeze(-1)).pow(2) * extras['weights']).sum(-1)
            # depth_std = depth_var.clamp(0., 1.).sqrt()
            # video_frame = np.concatenate((video_frame, cv2.applyColorMap(to8b(depth_std.cpu().numpy()), cv2.COLORMAP_VIRIDIS)), 1)

            cv2.imwrite(os.path.join(video_dir, str(img_idx) + '.png'), video_frame)        
            cv2.imwrite(os.path.join(video_depth_dir, str(img_idx) + '.png'), depth)
            cv2.imwrite(os.path.join(video_depth_colored_dir, str(img_idx) + '.png'), depth_colored_frame)

    ### Visualizing features
    pred_feats_res = pred_feats_res.detach().cpu().numpy()
    pred_feats_res = pred_feats_res.reshape((-1, args.feat_dim))
    
    ### This produces bad things because of the background --> this was not supervised
    # pca = PCA(n_components=4).fit(pred_feats_res)

    ### For Visualization ###
    N = features.shape[0]
    gt_features = features.detach().cpu().numpy()
    gt_features = gt_features.reshape((-1, args.feat_dim))
    pca = PCA(n_components=4).fit(gt_features)
    gt_pca_descriptors = pca.transform(gt_features)
    gt_pca_descriptors = gt_pca_descriptors.reshape((N, H, W, -1))
    comp_min = gt_pca_descriptors.min(axis=(0, 1, 2))[-3:]
    comp_max = gt_pca_descriptors.max(axis=(0, 1, 2))[-3:]
    #########################

    pred_pca_descriptors = pca.transform(pred_feats_res)
    pred_pca_descriptors = pred_pca_descriptors.reshape((len(idx_to_take), H, W, -1))

    for n in range(len(idx_to_take)):
      img_idx = idx_to_take[n]
      curr_feat = pred_pca_descriptors[n]
      pred_features = curr_feat[:, :, -3:]
   
      pred_features_img = (pred_features - comp_min) / (comp_max - comp_min)
      pred_features_pil = Image.fromarray((pred_features_img * 255).astype(np.uint8))
      pred_features_pil.save(os.path.join(video_feat_dir, str(img_idx) + '.png'))

    video_file = os.path.join(args.ckpt_dir, args.expname, filename + '.mp4')
    imgs = os.listdir(video_dir)
    imgs = natsorted(imgs)
    print(len(imgs))

    imageio.mimsave(video_file,
                    [imageio.imread(os.path.join(video_dir, img)) for img in imgs],
                    fps=10, macro_block_size=1)
    print("Done with " + video_file + ".")

    ## depth
    video_file = os.path.join(args.ckpt_dir, args.expname, filename + '_depth.mp4')
    imgs = os.listdir(video_depth_colored_dir)
    imgs = natsorted(imgs)
    print(len(imgs))

    imageio.mimsave(video_file,
                    [imageio.imread(os.path.join(video_depth_colored_dir, img)) for img in imgs],
                    fps=10, macro_block_size=1)
    print("Done with " + video_file + ".")

    ## feature
    video_file = os.path.join(args.ckpt_dir, args.expname, filename + '_feature.mp4')
    imgs = os.listdir(video_feat_dir)
    imgs = natsorted(imgs)
    print(len(imgs))

    imageio.mimsave(video_file,
                    [imageio.imread(os.path.join(video_feat_dir, img)) for img in imgs],
                    fps=10, macro_block_size=1)
    print("Done with " + video_file + ".")

    # video_file = os.path.join(args.ckpt_dir, args.expname, filename + '.mp4')
    # subprocess.call(["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(video_dir, "%d.jpg"), "-c:v", "libx264", "-profile:v", "high", "-crf", str(fps), video_file])
    # print("Maximal depth in video: {}".format(max_depth_in_video))

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
    embedcam_fn=None, with_test_time_optimization=False, features=None, mode="train"):
    far = render_kwargs_test['far']

    if count is None:
        # take all images in order
        count = len(indices)
        img_i = indices
    else:
        # take random images
        img_i = np.random.choice(indices, size=count, replace=False)

    rgbs_res = torch.empty(count, 3, H, W)
    rgbs0_res = torch.empty(count, 3, H, W)
    target_rgbs_res = torch.empty(count, 3, H, W)
    depths_res = torch.empty(count, 1, H, W)
    depths0_res = torch.empty(count, 1, H, W)
    target_depths_res = torch.empty(count, 1, H, W)
    target_valid_depths_res = torch.empty(count, 1, H, W, dtype=bool)

    pred_feats_res = torch.empty(count, H, W, args.feat_dim)
    gt_feats_res = torch.empty(count, H, W, args.feat_dim)
    
    mean_metrics = MeanTracker()
    mean_depth_metrics = MeanTracker() # track separately since they are not always available
    for n, img_idx in enumerate(img_i):
        print("Render image {}/{}".format(n + 1, count), end="")
        target = images[img_idx]
        target_depth = depths[img_idx]
        target_valid_depth = valid_depths[img_idx]
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]

        curr_features = features[img_idx]

        if args.feat_dim == 768:
            curr_features = curr_features.permute(2, 0, 1).unsqueeze(0).float()
            curr_features = F.interpolate(curr_features, size=(H, W), mode='bilinear').squeeze().permute(1,2,0)

        curr_features = curr_features.to(device)

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
                render_kwargs_test["embedded_cam"] = embedcam_fn[img_idx]
        
        with torch.no_grad():
            if args.feat_dim == 768:
                rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk //16), c2w=pose, **render_kwargs_test)
            else:
                if mode != "test":
                  rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 4), c2w=pose, **render_kwargs_test)
                else:
                  rgb, _, _, extras = render(H, W, intrinsic, chunk=(args.chunk // 16), c2w=pose, **render_kwargs_test)

            pred_features = extras["feature_map"]
            feature_loss = torch.norm(pred_features - curr_features, p=1, dim=-1)

            ## Only use foreground
            feature_loss = feature_loss * target_valid_depth
            feature_loss = torch.mean(feature_loss)
            
            # compute depth rmse
            depth_rmse = compute_rmse(extras['depth_map'][target_valid_depth], target_depth[:, :, 0][target_valid_depth])
            if not torch.isnan(depth_rmse):
                depth_metrics = {"depth_rmse" : depth_rmse.item()}
                mean_depth_metrics.add(depth_metrics)

            ### Fit LSTSQ for white balancing
            rgb_reshape = rgb.view(1, -1, 3)
            target_reshape = target.view(1, -1, 3)

            ## No intercept          
            X = torch.linalg.lstsq(rgb_reshape, target_reshape).solution
            rgb_reshape = rgb_reshape @ X
            rgb_reshape = rgb_reshape.view(rgb.shape)
            rgb = rgb_reshape
            
            # compute color metrics
            img_loss = img2mse(rgb, target)
            psnr = mse2psnr(img_loss)
            print("PSNR: {}".format(psnr))
            print("Feature loss: {}".format(feature_loss))
            rgb = torch.clamp(rgb, 0, 1)
            ssim = structural_similarity(rgb.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)
            lpips = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
            
            # store result
            rgbs_res[n] = rgb.clamp(0., 1.).permute(2, 0, 1).cpu()
            target_rgbs_res[n] = target.permute(2, 0, 1).cpu()
            depths_res[n] = (extras['depth_map'] / far).unsqueeze(0).cpu()
            target_depths_res[n] = (target_depth[:, :, 0] / far).unsqueeze(0).cpu()
            target_valid_depths_res[n] = target_valid_depth.unsqueeze(0).cpu()

            pred_feats_res[n] = pred_features.cpu()
            gt_feats_res[n] = curr_features.cpu()

            metrics = {"img_loss" : img_loss.item(), "psnr" : psnr.item(), "ssim" : ssim, "lpips" : lpips[0, 0, 0], 'feature_loss':feature_loss.item()}
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target)
                psnr0 = mse2psnr(img_loss0)
                depths0_res[n] = (extras['depth0'] / far).unsqueeze(0).cpu()
                rgbs0_res[n] = torch.clamp(extras['rgb0'], 0, 1).permute(2, 0, 1).cpu()

                pred_features = extras["feature_map_0"]
                feature_loss_0 = torch.norm(pred_features - curr_features, p=1, dim=-1)
                ## Only use foreground
                feature_loss_0 = feature_loss_0 * target_valid_depth
                feature_loss_0 = torch.mean(feature_loss_0)                

                metrics.update({"img_loss0" : img_loss0.item(), "psnr0" : psnr0.item(), "feature_loss_0": feature_loss_0.item()})
            mean_metrics.add(metrics)
    
    ### PCA on dino features for visualization
    pred_feats_res = pred_feats_res.detach().cpu().numpy()
    pred_feats_res = pred_feats_res.reshape((-1, args.feat_dim))

    gt_feats_res = gt_feats_res.detach().cpu().numpy()
    gt_feats_res = gt_feats_res.reshape((-1, args.feat_dim))

    pca = PCA(n_components=4).fit(gt_feats_res)
    pred_pca_descriptors = pca.transform(pred_feats_res)
    gt_pca_descriptors = pca.transform(gt_feats_res)

    pred_pca_descriptors = pred_pca_descriptors.reshape((count, H, W, -1))
    gt_pca_descriptors = gt_pca_descriptors.reshape((count, H, W, -1))

    res = { "rgbs" :  rgbs_res, "target_rgbs" : target_rgbs_res, "depths" : depths_res, "target_depths" : target_depths_res, \
        "target_valid_depths" : target_valid_depths_res, "pred_features" : pred_pca_descriptors, "gt_features": gt_pca_descriptors}
    if 'rgb0' in extras:
        res.update({"rgbs0" : rgbs0_res, "depths0" : depths0_res,})
    all_mean_metrics = MeanTracker()
    all_mean_metrics.add({**mean_metrics.as_dict(), **mean_depth_metrics.as_dict()})
    return all_mean_metrics, res

def write_images_with_metrics(images, mean_metrics, far, args, with_test_time_optimization=False):
    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images2_" + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id)
    os.makedirs(result_dir, exist_ok=True)

    comp_min = images["gt_features"].min(axis=(0, 1, 2))[-3:]
    comp_max = images["gt_features"].max(axis=(0, 1, 2))[-3:]

    for n, (rgb, depth, pred_features, gt_features) in enumerate(zip(images["rgbs"].permute(0, 2, 3, 1).cpu().numpy(), \
            images["depths"].permute(0, 2, 3, 1).cpu().numpy(), images["pred_features"], images["gt_features"])):

        # write rgb
        cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".jpg"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))
        # write depth
        cv2.imwrite(os.path.join(result_dir, str(n) + "_d" + ".png"), to16b(depth))

        pred_features = pred_features[:, :, -3:]
        gt_features = gt_features[:, :, -3:]

        pred_features_img = (pred_features - comp_min) / (comp_max - comp_min)
        gt_features_img = (gt_features - comp_min) / (comp_max - comp_min)
        pred_features_pil = Image.fromarray((pred_features_img * 255).astype(np.uint8))
        gt_features_pil = Image.fromarray((gt_features_img * 255).astype(np.uint8))
        pred_features_pil.save(os.path.join(result_dir, str(n) + "_fpred" + ".png"))
        gt_features_pil.save(os.path.join(result_dir, str(n) + "_gtpred" + ".png"))

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

    model = NeRF_semantics(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs, semantic_dim = args.feat_dim)

    model = nn.DataParallel(model).to(device)
    grad_vars = list(model.parameters())

    grad_vars = []
    grad_names = []

    seman_grad_vars = []
    seman_grad_names = []

    for name, param in model.named_parameters():
        if "semantic" in name:
          seman_grad_vars.append(param)
          seman_grad_names.append(name)
        else:
          grad_vars.append(param)
          grad_names.append(name)

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF_semantics(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs, semantic_dim = args.feat_dim)
            
        model_fine = nn.DataParallel(model_fine).to(device)

        for name, param in model_fine.named_parameters():
            if "semantic" in name:
              seman_grad_vars.append(param)
              seman_grad_names.append(name)
            else:
              grad_vars.append(param)
              grad_names.append(name)

    network_query_fn = lambda inputs, viewdirs, embedded_cam, network_fn : run_network(inputs, viewdirs, embedded_cam, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                bb_center=args.bb_center,
                                                                bb_scale=args.bb_scale,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    print("Using different learning rate for feature layers.")
    optimizer.add_param_group({"params": seman_grad_vars, "lr":args.seman_lrate})

    start = 0

    ##########################

    # Load checkpoints
    ckpt = load_checkpoint(args)
    if ckpt is not None:
        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

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
    }
    render_kwargs_train.update(scene_render_params)

    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, grad_names

####################################################################
### Modifying this to create two nerf models to jointly optimize ###
####################################################################
def create_nerf2(args, scene_render_params):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    #####################
    ####### Shared ######
    #####################
    network_query_fn = lambda inputs, viewdirs, embedded_cam, network_fn : run_network(inputs, viewdirs, embedded_cam, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                bb_center=args.bb_center,
                                                                bb_scale=args.bb_scale,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)

    motion_network_query_fn = lambda inputs, features, network_fn : run_motion_potential(inputs, features, network_fn,
                                                                bb_center=args.bb_center,
                                                                bb_scale=args.bb_scale,
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus)
    grad_vars = []
    grad_names = []

    seman_grad_vars = []
    seman_grad_names = []

    motion_vars = []
    ####################

    ####### First NeRF model ######
    model1 = NeRF_semantics(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs, semantic_dim = args.feat_dim)

    model1 = nn.DataParallel(model1).to(device)

    for name, param in model1.named_parameters():
        if "semantic" in name:
          seman_grad_vars.append(param)
          seman_grad_names.append(name)
        else:
          grad_vars.append(param)
          grad_names.append(name)

    model_fine1 = None
    if args.N_importance > 0:
        model_fine1 = NeRF_semantics(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs, semantic_dim = args.feat_dim)
            
        model_fine1 = nn.DataParallel(model_fine1).to(device)

        for name, param in model_fine1.named_parameters():
            if "semantic" in name:
              seman_grad_vars.append(param)
              seman_grad_names.append(name)
            else:
              grad_vars.append(param)
              grad_names.append(name)

    ###### Load pretrained model #####
    path1 = os.path.join(args.pretrained_dir, args.pretrained_fol1)
    ckpts = [os.path.join(path1, f) for f in sorted(os.listdir(path1)) if '000.tar' in f]
    print('Found ckpts', ckpts)
    print("Loading pretrained model 1....")
    ckpt = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

    if ckpt is not None:
        # Load model
        model1.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine1 is not None:
            model_fine1.load_state_dict(ckpt['network_fine_state_dict'])

    ### Motion model
    motion_model1 = MotionPotential(input_ch=3, input_ch_feature=args.feat_dim, output_ch=3)
    motion_model1 = nn.DataParallel(motion_model1).to(device)

    for name, param in motion_model1.named_parameters():
        motion_vars.append(param)
    ###################################

    ####### Second NeRF model ######
    model2 = NeRF_semantics(D=args.netdepth, W=args.netwidth,
                  input_ch=input_ch, output_ch=output_ch, skips=skips,
                  input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs, semantic_dim = args.feat_dim)

    model2 = nn.DataParallel(model2).to(device)

    for name, param in model2.named_parameters():
        if "semantic" in name:
          seman_grad_vars.append(param)
          seman_grad_names.append(name)
        else:
          grad_vars.append(param)
          grad_names.append(name)

    model_fine2 = None
    if args.N_importance > 0:
        model_fine2 = NeRF_semantics(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs, semantic_dim = args.feat_dim)
            
        model_fine2 = nn.DataParallel(model_fine2).to(device)

        for name, param in model_fine2.named_parameters():
            if "semantic" in name:
              seman_grad_vars.append(param)
              seman_grad_names.append(name)
            else:
              grad_vars.append(param)
              grad_names.append(name)

    ###### Load pretrained model #####
    path2 = os.path.join(args.pretrained_dir, args.pretrained_fol2)
    ckpts = [os.path.join(path2, f) for f in sorted(os.listdir(path2)) if '000.tar' in f]
    print('Found ckpts', ckpts)
    print("Loading pretrained model 2....")
    ckpt = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

    if ckpt is not None:
        # Load model
        model2.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine2 is not None:
            model_fine2.load_state_dict(ckpt['network_fine_state_dict'])

    ### Motion model
    motion_model2 = MotionPotential(input_ch=3, input_ch_feature=args.feat_dim, output_ch=3)
    motion_model2 = nn.DataParallel(motion_model2).to(device)

    for name, param in motion_model2.named_parameters():
        motion_vars.append(param)
    ################################

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    print("Using different learning rate for feature layers.")
    optimizer.add_param_group({"params": seman_grad_vars, "lr":args.seman_lrate})

    optimizer_motion = torch.optim.Adam(params=motion_vars, lr=args.motion_lrate, betas=(0.9, 0.999))

    start = 0
    ##########################

    ### Load checkpoints
    ckpt = load_checkpoint(args)
    if ckpt is not None:
        start = ckpt['global_step']

        # Load model
        model1.load_state_dict(ckpt['network_fn1_state_dict'])
        if model_fine1 is not None:
            model_fine1.load_state_dict(ckpt['network_fine1_state_dict'])
        model2.load_state_dict(ckpt['network_fn2_state_dict'])
        if model_fine2 is not None:
            model_fine2.load_state_dict(ckpt['network_fine2_state_dict'])

        motion_model1.load_state_dict(ckpt['network_motion1_state_dict'])
        motion_model2.load_state_dict(ckpt['network_motion2_state_dict'])


    ##########################
    embedded_cam = torch.tensor((), device=device)
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'motion_network_query_fn' : motion_network_query_fn,
        'embedded_cam' : embedded_cam,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine1' : model_fine1,
        'network_fine2' : model_fine2,
        'N_samples' : args.N_samples,
        'network_fn1' : model1,
        'network_fn2' : model2,
        'network_motion1' : motion_model1,
        'network_motion2' : motion_model2,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
    }
    render_kwargs_train.update(scene_render_params)

    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, grad_names, optimizer_motion
####################################################################


def compute_weights(raw, z_vals, rays_d, noise=0.):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    return weights

def raw2depth(raw, z_vals, rays_d):
    weights = compute_weights(raw, z_vals, rays_d)
    depth = torch.sum(weights * z_vals, -1)
    std = (((z_vals - depth.unsqueeze(-1)).pow(2) * weights).sum(-1)).sqrt()
    return depth, std

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
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
    features = raw[...,4:]

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    weights = compute_weights(raw, z_vals, rays_d, noise)
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    feature_map = torch.sum(weights[...,None].detach() * features, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map, feature_map

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

def render_rays(ray_batch,
                use_viewdirs,
                network_fn1,
                network_fn2,
                network_query_fn,
                N_samples,
                precomputed_z_samples=None,
                embedded_cam=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine1=None,
                network_fine2=None,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                is_joint=False,
                cached_u= None,
                idx = 0,
                network_motion1 = None,
                network_motion2 = None,
                motion_network_query_fn = None,            
                for_motion=False):
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
    t_vals = torch.linspace(0., 1., steps=N_samples)

    # sample and render rays for dense depth priors for nerf
    N_samples_half = N_samples // 2
    
    # sample and render rays for nerf
    if not lindisp:
        # print("Not lindisp")
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # print("Lindisp")
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb > 0.:
        # print("Perturb.")
        z_vals = perturb_z_vals(z_vals, pytest)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    if idx == 0:
      raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn1)
    else:
      raw = network_query_fn(pts, viewdirs, embedded_cam, network_fn2)

    rgb_map, disp_map, acc_map, weights, depth_map, feature_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

    ### Try without coarse and fine network, but just one network and use additional samples from the distribution of the nerf
    if N_importance == 0:

        ### P_depth from base network
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            z_vals_2 = sample_pdf(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)
        else:
            z_vals_2 = sample_pdf_joint(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)
        #########################

        ### Forward the rendering network with the additional samples
        pts_2 = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_2[...,:,None]
        
        if idx == 0:
          raw_2 = network_query_fn(pts_2, viewdirs, embedded_cam, network_fn1)
        else:
          raw_2 = network_query_fn(pts_2, viewdirs, embedded_cam, network_fn2) 

        z_vals = torch.cat((z_vals, z_vals_2), -1)
        raw = torch.cat((raw, raw_2), 1)
        z_vals, indices = z_vals.sort()

        ### Concatenated output
        raw = torch.gather(raw, 1, indices.unsqueeze(-1).expand_as(raw))
        rgb_map, disp_map, acc_map, weights, depth_map, feature_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)


        ## Second tier P_depth
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            z_vals_output = sample_pdf(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)
        else:
            z_vals_output = sample_pdf_joint(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)

        pred_depth_hyp = torch.cat((z_vals_2, z_vals_output), -1)


    elif N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, z_vals_0, weights_0, feature_map_0 = rgb_map, disp_map, acc_map, depth_map, z_vals, weights, feature_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        ## Original NeRF uses this
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        
        ## To model p_depth from coarse network
        z_samples_depth = torch.clone(z_samples)

        ## For fine network sampling
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        # run_fn = network_fn2 if network_fine2 is None else network_fine2
        # raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)

        if idx == 0:
          run_fn = network_fn1 if network_fine1 is None else network_fine1
          raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)
        else:
          run_fn = network_fn2 if network_fine2 is None else network_fine2
          raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, feature_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

        ### P_depth from fine network
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            z_samples, u = sample_pdf_return_u(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u)
        else:
            z_samples, u = sample_pdf_joint_return_u(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u)

        pred_depth_hyp = z_samples

    #### For unary energies ####    
    if for_motion:
      num_samples_for_database = 50
      if not is_joint:
          z_vals_importance, u = sample_pdf_return_u(z_vals_mid, weights[...,1:-1], num_samples_for_database, det=(perturb==0.), pytest=pytest, load_u=cached_u)
      else:
          z_vals_importance, u = sample_pdf_joint_return_u(z_vals_mid, weights[...,1:-1], num_samples_for_database, det=(perturb==0.), pytest=pytest, load_u=cached_u)

      z_vals_importance, _ = torch.sort(z_vals_importance.detach(), -1)
      pnm_pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_importance[...,:,None] # [N_rays, N_samples + N_importance, 3]

      raw_importance = network_query_fn(pnm_pts, viewdirs, embedded_cam, run_fn)
      rgb_map_pnm, _, _, weights_pnm, _, _ = raw2outputs(raw_importance, z_vals_importance, rays_d, raw_noise_std, pytest=pytest)

      ### Get color and features for caching of motion database
      pnm_rgb_term = torch.sigmoid(raw_importance[...,:3])

      pnm_feature_term = raw_importance[...,4:]
    #############################

    if not for_motion:
      ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights, 'pred_hyp' : pred_depth_hyp,\
      'u':u, 'feature_map': feature_map}
      if retraw:
          ret['raw'] = raw
      if N_importance > 0:
          ret['rgb0'] = rgb_map_0
          ret['disp0'] = disp_map_0
          ret['acc0'] = acc_map_0
          ret['depth0'] = depth_map_0
          ret['z_vals0'] = z_vals_0
          ret['weights0'] = weights_0
          ret['feature_map_0'] = feature_map_0
          ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
          # ret['pred_hyp'] = pred_depth_hyp

      for k in ret:
          if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
              print(f"! [Numerical Error] {k} contains nan or inf.")
    
    else:
      ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'pnm_rgb_term': pnm_rgb_term, 'pnm_feature_term': pnm_feature_term, 'pnm_points': pnm_pts, \
      "rgb_map_pnm": rgb_map_pnm, "pnm_weights": weights_pnm}

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

    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    return batch_rays, target_s, target_d, target_vd, img_i

def get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, intrinsics, all_hypothesis, curr_features, args, space_carving_idx=None, cached_u=None, gt_valid_depths=None):
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
    target_feat = curr_features[select_coords[:, 0], select_coords[:, 1]] 
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

    if gt_valid_depths is not None:
        space_carving_mask = gt_valid_depths[img_i].squeeze()
        space_carving_mask = space_carving_mask[select_coords[:, 0], select_coords[:, 1]]
    else:
        space_carving_mask = None

    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    
    return batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u, target_feat


def train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis, is_init_scales=False, \
              scales_init=None, shifts_init=None, use_depth=False, features_fnames=None, features=None):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    tb = SummaryWriter(log_dir=os.path.join("runs", args.expname))
    near, far = scene_sample_params['near'], scene_sample_params['far']
    H, W = images.shape[2:4]
    i_train, i_test = i_split
    i_val = i_test
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)
    print('TEST views are', i_test)

    test_images = images
    test_poses = poses
    test_intrinsics = intrinsics
    i_test = i_test

    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    intrinsics = torch.Tensor(intrinsics).to(device)

    # if use_depth:
    #     if args.dataset != "blender":
    #         depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], 1)).to(device)
    #         valid_depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2]), dtype=bool).to(device)
    #         all_depth_hypothesis = torch.Tensor(all_depth_hypothesis).to(device)
    #     else:
    #         depths = torch.Tensor(depths).to(device)
    #         valid_depths = torch.Tensor(valid_depths).bool().to(device)            
    #         gt_depths_train = depths.unsqueeze(1)
    #         gt_valid_depths_train = valid_depths.unsqueeze(1)

    # else:        
    #     depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], 1)).to(device)
    #     valid_depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2]), dtype=bool).to(device)

    if use_depth:
        if args.dataset != "blender":
            depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], images.shape[3], 1)).to(device)
            valid_depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], images.shape[3]), dtype=bool).to(device)
            all_depth_hypothesis = torch.Tensor(all_depth_hypothesis).to(device)
        else:
            depths = torch.Tensor(depths).to(device)
            valid_depths = torch.Tensor(valid_depths).bool().to(device)            
            gt_depths_train = depths.unsqueeze(2)
            gt_valid_depths_train = valid_depths.unsqueeze(2)

    else:        
        depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], images.shape[3], 1)).to(device)
        valid_depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], images.shape[3]), dtype=bool).to(device)


    # create nerf model
    render_kwargs_train, render_kwargs_test, start, nerf_grad_vars, optimizer, nerf_grad_names, optimizer_motion = create_nerf2(args, scene_sample_params)
    print("Loaded models.")

    # if use_depth:
    #     ##### Initialize depth scale and shift
    #     DEPTH_SCALES = torch.autograd.Variable(torch.ones((images.shape[0], images.shape[1], 1), dtype=torch.float, device=images.device)*args.scale_init, requires_grad=True)
    #     DEPTH_SHIFTS = torch.autograd.Variable(torch.ones((images.shape[0], images.shape[1], 1), dtype=torch.float, device=images.device)*args.shift_init, requires_grad=True)

    #     print(DEPTH_SCALES)
    #     print()
    #     print(DEPTH_SHIFTS)
    #     print()
    #     print(DEPTH_SCALES.shape)
    #     print(DEPTH_SHIFTS.shape)

    #     optimizer_ss = torch.optim.Adam(params=(DEPTH_SCALES, DEPTH_SHIFTS,), lr=args.scaleshift_lr)
        
    #     print("Initialized scale and shift.")
    #     ################################

    # create camera embedding function
    embedcam_fn = None

    # optimize nerf
    print('Begin')
    N_iters = args.num_iterations + 1
    global_step = start
    start = start + 1

    init_learning_rate = args.lrate
    old_learning_rate = init_learning_rate     
    ALL_DATABASE = None

    skip_view = args.skip_views
    H_database = H
    W_database = W

    if IS_MOTION_DEBUG:
      ### For debugging feture visualization ###
      
      ### Only get the first frame's
      N = features.shape[1]
      gt_features = features[0].detach().cpu().numpy()

      gt_features = gt_features.reshape((-1, args.feat_dim))
      pca = PCA(n_components=4).fit(gt_features)
      gt_pca_descriptors = pca.transform(gt_features)
      gt_pca_descriptors = gt_pca_descriptors.reshape((N, H, W, -1))
      comp_min = gt_pca_descriptors.min(axis=(0, 1, 2))[-3:]
      comp_max = gt_pca_descriptors.max(axis=(0, 1, 2))[-3:]
      ##########################################

    #############################
    ##### Computing for STD #####
    #############################

    print("Computing STD for scaling of dimensions.")
    SCALE_FACTORS = []

    with torch.no_grad():
      DATABASE1 = []
      DATABASE2 = []

      for idx in range(0, len(i_train), skip_view):
        img_i = i_train[idx]      

        #### Downsample to get a smaller size
        curr_valid_depth1 = valid_depths[0][img_i]
        curr_valid_depth1 = curr_valid_depth1.unsqueeze(-1).permute(2, 0, 1).unsqueeze(0).float()
        curr_valid_depth1 = F.interpolate(curr_valid_depth1, size=(H_database, W_database), mode='nearest').squeeze().bool()
        is_object_ray1 = curr_valid_depth1 

        curr_valid_depth2 = valid_depths[1][img_i]
        curr_valid_depth2 = curr_valid_depth2.unsqueeze(-1).permute(2, 0, 1).unsqueeze(0).float()
        curr_valid_depth2 = F.interpolate(curr_valid_depth2, size=(H_database, W_database), mode='nearest').squeeze().bool()
        is_object_ray2 = curr_valid_depth2 

        ### Sampled for unary potentials
        _, _, _, extras1_unary = render(H_database, W_database, intrinsics[0][img_i], chunk=(args.chunk // 8), c2w=poses[0][img_i][:3,:4], with_5_9=False, idx=0, for_motion=True, **render_kwargs_train)
        _, _, _, extras2_unary = render(H_database, W_database, intrinsics[1][img_i], chunk=(args.chunk // 8), c2w=poses[1][img_i][:3,:4], with_5_9=False, idx=1, for_motion=True, **render_kwargs_train)        

        pnm_rgb_term1, pnm_feature_term1, pnm_points1, pnm_rgb_map1 = extras1_unary["pnm_rgb_term"], extras1_unary["pnm_feature_term"], extras1_unary["pnm_points"], extras1_unary["rgb_map_pnm"]
        pnm_rgb_term2, pnm_feature_term2, pnm_points2, pnm_rgb_map2 = extras2_unary["pnm_rgb_term"], extras2_unary["pnm_feature_term"], extras2_unary["pnm_points"], extras2_unary["rgb_map_pnm"]

        # ### This is selecting the sample with the max weight
        # pnm_weights1 = extras1_unary["pnm_weights"]
        # pnm_weights2 = extras2_unary["pnm_weights"]
        # print(pnm_weights1.shape)
        # print(pnm_weights2.shape)

        # best_pnm_index1 = torch.argmin(pnm_weights1, dim=-1)
        # best_pnm_index2 = torch.argmin(pnm_weights2, dim=-1)

        # ### Torch gather
        # print(best_pnm_index1.shape)
        # print(best_pnm_index2.shape)
        # pnm_rgb_term1 = torch.gather(pnm_rgb_term1, -2, best_pnm_index1.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).squeeze()
        # pnm_feature_term1 = torch.gather(pnm_feature_term1, -2, best_pnm_index1.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1, args.feat_dim)).squeeze()
        # pnm_points1 = torch.gather(pnm_points1, -2, best_pnm_index1.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).squeeze()

        # pnm_rgb_term2 = torch.gather(pnm_rgb_term2, -2, best_pnm_index2.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).squeeze()
        # pnm_feature_term2 = torch.gather(pnm_feature_term2, -2, best_pnm_index2.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1, args.feat_dim)).squeeze()
        # pnm_points2 = torch.gather(pnm_points2, -2, best_pnm_index2.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)).squeeze()

        ### This is taking the average
        pnm_rgb_term1 = torch.mean(pnm_rgb_term1, axis=-2)
        pnm_rgb_term2 = torch.mean(pnm_rgb_term2, axis=-2)
        pnm_feature_term1 = torch.mean(pnm_feature_term1, axis=-2)
        pnm_feature_term2 = torch.mean(pnm_feature_term2, axis=-2)
        pnm_points1 = torch.mean(pnm_points1, axis=-2)
        pnm_points2 = torch.mean(pnm_points2, axis=-2)

        if IS_MOTION_DEBUG:
          # print(pnm_rgb_term1.shape)
          # print(pnm_feature_term1.shape)
          # print(pnm_points1.shape)
          # print(pnm_rgb_map1.shape)
          # print()
          # print(pnm_rgb_term2.shape)
          # print(pnm_feature_term2.shape)
          # print(pnm_points2.shape)
          # print(pnm_rgb_map2.shape)
          # exit()

          cv2.imwrite("testviz5_pnm1.jpg", cv2.cvtColor(to8b(pnm_rgb_term1.detach().cpu().numpy()), cv2.COLOR_RGB2BGR))
          cv2.imwrite("testviz5_pnm2.jpg", cv2.cvtColor(to8b(pnm_rgb_term2.detach().cpu().numpy()), cv2.COLOR_RGB2BGR))
          cv2.imwrite("testviz5_pnmrgb1.jpg", cv2.cvtColor(to8b(pnm_rgb_map1.detach().cpu().numpy()), cv2.COLOR_RGB2BGR))
          cv2.imwrite("testviz5_pnmrgb2.jpg", cv2.cvtColor(to8b(pnm_rgb_map2.detach().cpu().numpy()), cv2.COLOR_RGB2BGR))

          ### Feature
          pred_pca_descriptors = pca.transform(pnm_feature_term1.detach().cpu().numpy().reshape((-1, args.feat_dim)))
          pred_pca_descriptors = pred_pca_descriptors.reshape((H, W, -1)) 
          pred_features = pred_pca_descriptors[:, :, -3:]
          pred_features_img = (pred_features - comp_min) / (comp_max - comp_min)
          pred_features_pil = Image.fromarray((pred_features_img * 255).astype(np.uint8))
          pred_features_pil.save('testviz5_feature1.png')

          pred_pca_descriptors = pca.transform(pnm_feature_term2.detach().cpu().numpy().reshape((-1, args.feat_dim)))
          pred_pca_descriptors = pred_pca_descriptors.reshape((H, W, -1)) 
          pred_features = pred_pca_descriptors[:, :, -3:]
          pred_features_img = (pred_features - comp_min) / (comp_max - comp_min)
          pred_features_pil = Image.fromarray((pred_features_img * 255).astype(np.uint8))
          pred_features_pil.save('testviz5_feature2.png')

          ### 3D points
          pnm_points1_ = pnm_points1[is_object_ray1].reshape(-1, 3)
          pts1 = pnm_points1_.detach().cpu().numpy().reshape(-1, 3)
          colors1 = pnm_rgb_term1[is_object_ray1].detach().cpu().numpy()
          save_pointcloud_samples(pts1, colors1, "testviz5_rgb1.png")

          # print(colors1.shape)
          # c = cv2.cvtColor(to8b(colors1), cv2.COLOR_RGB2BGR)
          # print(c.shape)
          # print(to8b(colors1).shape)
          # exit()
          save_point_cloud(pts1, to8b(colors1), 'testviz5_rgb1.ply')
          print(torch.max(pnm_rgb_term1[is_object_ray1]))

          pnm_points2_ = pnm_points2[is_object_ray2].reshape(-1, 3)
          pts2 = pnm_points2_.detach().cpu().numpy().reshape(-1, 3)
          colors2 = pnm_rgb_term2[is_object_ray2].detach().cpu().numpy()
          save_pointcloud_samples(pts2, colors2, "testviz5_rgb2.png")

          # c = cv2.cvtColor(to8b(colors2), cv2.COLOR_RGB2BGR)

          save_point_cloud(pts2, to8b(colors2), 'testviz5_rgb2.ply')
          print(torch.max(pnm_rgb_term2[is_object_ray2]))

          # ####
          # vect1 = np.zeros(pts1.shape)
          # vect1[..., 0] = 0.
          # vect1[..., 1] = 0.25
          # vect1[..., 2] = 0.0

          # vect2 = np.zeros(pts2.shape)
          # vect2[..., 0] = 0.
          # vect2[..., 1] = 0.25
          # vect2[..., 2] = 0.0

          # save_motion_vectors(pts1, colors1, vect1, "testviz4_withlines1.png")
          # save_motion_vectors(pts2, colors2, vect2, "testviz4_withlines2.png")
        
        #### Construct database elements ####
        ## Database 1
        pnm_rgb_term1 = pnm_rgb_term1[is_object_ray1].reshape(-1, 3)
        pnm_feature_term1 = pnm_feature_term1[is_object_ray1].reshape(-1, args.feat_dim)
        pnm_points1 = pnm_points1[is_object_ray1].reshape(-1, 3)

        motion_query_func = render_kwargs_train["motion_network_query_fn"]
        motion_model1 = render_kwargs_train["network_motion1"]

        potentials1 = motion_query_func(pnm_points1, pnm_feature_term1*0.5, motion_model1)

        pnm_rgb_term2 = pnm_rgb_term2[is_object_ray2].reshape(-1, 3)
        pnm_feature_term2 = pnm_feature_term2[is_object_ray2].reshape(-1, args.feat_dim)
        pnm_points2 = pnm_points2[is_object_ray2].reshape(-1, 3)

        motion_query_func = render_kwargs_train["motion_network_query_fn"]
        motion_model2 = render_kwargs_train["network_motion2"]
        
        potentials2 = motion_query_func(pnm_points2, pnm_feature_term2*0.5, motion_model2)
        
        # ### DEBUG --- Overwrite to set potentials to a constant
        # potentials1 = torch.ones(potentials1.shape).to(potentials1.device) * 1.5
        # potentials2 = torch.ones(potentials2.shape).to(potentials2.device)

        if IS_MOTION_DEBUG:
          print(potentials1)
          print(potentials1.shape)
          print(potentials2)
          print(potentials2.shape)

          ### Test smoothness of potential field
          pts1_dummy = torch.zeros(pnm_points1.shape).to(pnm_points1.device)
          feature1_dummy = torch.ones(pnm_feature_term1.shape).to(pnm_feature_term1.device)
          potentials1_ptsdummy = motion_query_func(pts1_dummy, pnm_feature_term1, motion_model1)
          potentials1_featuredummy = motion_query_func(pnm_points1, feature1_dummy, motion_model1)

          pts2_dummy = torch.zeros(pnm_points2.shape).to(pnm_points2.device)
          feature2_dummy = torch.ones(pnm_feature_term2.shape).to(pnm_feature_term2.device)
          potentials2_ptsdummy = motion_query_func(pts2_dummy, pnm_feature_term2, motion_model2)
          potentials2_featuredummy = motion_query_func(pnm_points2, feature2_dummy, motion_model2)

          ### Visualize potential initialization
          pts1 = pnm_points1.detach().cpu().numpy().reshape(-1, 3)
          colors1 = pnm_rgb_term1.detach().cpu().numpy()
          save_motion_vectors(pts1, colors1, potentials1.detach().cpu().numpy(), "testviz6_init_potentials1_s05.png")
          save_motion_vectors(pts1, colors1, potentials1.detach().cpu().numpy(), "testviz6_init_potentials1_endpt_s05.png", is_vec=False)

          # save_motion_vectors(pts1, colors1, potentials1_ptsdummy.detach().cpu().numpy(), "testviz5_init_potentials1_ptsdummy.png")
          # save_motion_vectors(pts1, colors1, potentials1_ptsdummy.detach().cpu().numpy(), "testviz5_init_potentials1_endpt_ptsdummy.png", is_vec=False)
          # save_motion_vectors(pts1, colors1, potentials1_featuredummy.detach().cpu().numpy(), "testviz5_init_potentials1_featuredummy.png")
          # save_motion_vectors(pts1, colors1, potentials1_featuredummy.detach().cpu().numpy(), "testviz5_init_potentials1_endpt_featuredummy.png", is_vec=False)

          pts2 = pnm_points2.detach().cpu().numpy().reshape(-1, 3)
          colors2 = pnm_rgb_term2.detach().cpu().numpy()
          save_motion_vectors(pts2, colors2, potentials2.detach().cpu().numpy(), "testviz6_init_potentials2_s05.png")
          save_motion_vectors(pts2, colors2, potentials2.detach().cpu().numpy(), "testviz6_init_potentials2_endpt_s05.png", is_vec=False)

          # save_motion_vectors(pts2, colors2, potentials2_ptsdummy.detach().cpu().numpy(), "testviz5_init_potentials2_ptsdummy.png")
          # save_motion_vectors(pts2, colors2, potentials2_ptsdummy.detach().cpu().numpy(), "testviz5_init_potentials2_endpt_ptsdummy.png", is_vec=False)
          # save_motion_vectors(pts2, colors2, potentials2_featuredummy.detach().cpu().numpy(), "testviz5_init_potentials2_featuredummy.png")
          # save_motion_vectors(pts2, colors2, potentials2_featuredummy.detach().cpu().numpy(), "testviz5_init_potentials2_endpt_featuredummy.png", is_vec=False)

          exit()


        curr_entries1 = torch.cat([pnm_rgb_term1, pnm_feature_term1, potentials1 - pnm_points1], -1)
        curr_entries2 = torch.cat([pnm_rgb_term2, pnm_feature_term2, potentials2 - pnm_points2], -1)

        # print(curr_entries1.shape)
        # print(curr_entries2.shape)

        DATABASE1.append(curr_entries1)
        DATABASE2.append(curr_entries2)
        #####################################

      DATABASE1 = torch.cat(DATABASE1, 0)
      curr_pnm_rgb_term1, curr_pnm_feature_term1, curr_potentials_term1 = torch.split(DATABASE1, [3, args.feat_dim,3], dim=-1)
      std_rgb1 = torch.std(curr_pnm_rgb_term1, dim=0)
      std_feature1 = torch.std(curr_pnm_feature_term1, dim=0)
      std_potentials1 = torch.std(curr_potentials_term1, dim=0)
      std_rgb1 = torch.sum(std_rgb1)
      std_feature1 = torch.sum(std_feature1)
      std_potentials1 = torch.sum(std_potentials1)

      weight_rgb1 = (1./std_rgb1).float()
      weight_features1 = (1./std_feature1).float()
      weight_potentials1 = (1./std_potentials1).float()
      SCALE_FACTORS.append(np.array([weight_rgb1.detach().cpu().numpy(), weight_features1.detach().cpu().numpy(), weight_potentials1.detach().cpu().numpy()]))

      # print(std_rgb1)
      # print(std_feature1)
      # print(std_potentials1)
      # print()

      DATABASE2 = torch.cat(DATABASE2, 0)
      curr_pnm_rgb_term2, curr_pnm_feature_term2, curr_potentials_term2 = torch.split(DATABASE2, [3, args.feat_dim,3], dim=-1)
      std_rgb2 = torch.std(curr_pnm_rgb_term2, dim=0)
      std_feature2 = torch.std(curr_pnm_feature_term2, dim=0)
      std_potentials2 = torch.std(curr_potentials_term2, dim=0)
      std_rgb2 = torch.sum(std_rgb2)
      std_feature2 = torch.sum(std_feature2)
      std_potentials2 = torch.sum(std_potentials2)

      weight_rgb2 = (1./std_rgb2).float()
      weight_features2 = (1./std_feature2).float()
      weight_potentials2 = (1./std_potentials2).float()
      SCALE_FACTORS.append(np.array([weight_rgb2.detach().cpu().numpy(), weight_features2.detach().cpu().numpy(), weight_potentials2.detach().cpu().numpy()]))

      # print(std_rgb2)
      # print(std_feature2)
      # print(std_potentials2)
      # print()
      # exit()

      print("Computed scales for energies:")
      print(SCALE_FACTORS)

    #### Save scale factors ###
    output_dict = {"frame1": SCALE_FACTORS[0], "frame2": SCALE_FACTORS[1]}
    fname = os.path.join(args.ckpt_dir, args.expname, "database_scales.npy")
    np.save(fname, output_dict)
    exit()

    # #### Debug load scales
    # input_dict = np.load(fname, allow_pickle=True)
    # scales1 = input_dict.item().get('frame1')
    # scales2 = input_dict.item().get('frame2')

    # print(scales1)
    # print(scales2)
    # print()

    # weight_rgb1, weight_features1, weight_potentials1 = SCALE_FACTORS[0]
    # print(weight_rgb1)
    # print(weight_features1)
    # print(weight_potentials1)
    # exit()

    #############################
    

    ####################################
    ##### Training Motion Potential ####
    ####################################

    for i in trange(start, N_iters):

      DATABASE1 = []
      DATABASE2 = []

      #######################################
      ###### Constructing the Database ######
      #######################################

      for idx in range(0, len(i_train), skip_view):
        img_i = i_train[idx]      

        #### Downsample to get a smaller size
        curr_valid_depth1 = valid_depths[0][img_i]
        curr_valid_depth1 = curr_valid_depth1.unsqueeze(-1).permute(2, 0, 1).unsqueeze(0).float()
        curr_valid_depth1 = F.interpolate(curr_valid_depth1, size=(H_database, W_database), mode='nearest').squeeze().bool()
        is_object_ray1 = curr_valid_depth1 

        curr_valid_depth2 = valid_depths[1][img_i]
        curr_valid_depth2 = curr_valid_depth2.unsqueeze(-1).permute(2, 0, 1).unsqueeze(0).float()
        curr_valid_depth2 = F.interpolate(curr_valid_depth2, size=(H_database, W_database), mode='nearest').squeeze().bool()
        is_object_ray2 = curr_valid_depth2 

        with torch.no_grad():
          ### Sampled for unary potentials
          _, _, _, extras1_unary = render(H_database, W_database, intrinsics[0][img_i], chunk=(args.chunk // 8), c2w=poses[0][img_i][:3,:4], with_5_9=False, idx=0, for_motion=True, **render_kwargs_train)
          _, _, _, extras2_unary = render(H_database, W_database, intrinsics[1][img_i], chunk=(args.chunk // 8), c2w=poses[1][img_i][:3,:4], with_5_9=False, idx=1, for_motion=True, **render_kwargs_train)        

          pnm_rgb_term1, pnm_feature_term1, pnm_points1, pnm_rgb_map1 = extras1_unary["pnm_rgb_term"], extras1_unary["pnm_feature_term"], extras1_unary["pnm_points"], extras1_unary["rgb_map_pnm"]
          pnm_rgb_term2, pnm_feature_term2, pnm_points2, pnm_rgb_map2 = extras2_unary["pnm_rgb_term"], extras2_unary["pnm_feature_term"], extras2_unary["pnm_points"], extras2_unary["rgb_map_pnm"]

          ### This is taking the average
          pnm_rgb_term1 = torch.mean(pnm_rgb_term1, axis=-2)
          pnm_rgb_term2 = torch.mean(pnm_rgb_term2, axis=-2)
          pnm_feature_term1 = torch.mean(pnm_feature_term1, axis=-2)
          pnm_feature_term2 = torch.mean(pnm_feature_term2, axis=-2)
          pnm_points1 = torch.mean(pnm_points1, axis=-2)
          pnm_points2 = torch.mean(pnm_points2, axis=-2)
        
        #### Motion Potentials ####
        ## Database 1
        pnm_rgb_term1 = pnm_rgb_term1[is_object_ray1].reshape(-1, 3)
        pnm_feature_term1 = pnm_feature_term1[is_object_ray1].reshape(-1, args.feat_dim)
        pnm_points1 = pnm_points1[is_object_ray1].reshape(-1, 3)

        motion_query_func = render_kwargs_train["motion_network_query_fn"]
        motion_model1 = render_kwargs_train["network_motion1"]
        potentials1 = motion_query_func(pnm_points1, pnm_feature_term1, motion_model1)

        ## Database 2
        pnm_rgb_term2 = pnm_rgb_term2[is_object_ray2].reshape(-1, 3)
        pnm_feature_term2 = pnm_feature_term2[is_object_ray2].reshape(-1, args.feat_dim)
        pnm_points2 = pnm_points2[is_object_ray2].reshape(-1, 3)

        motion_query_func = render_kwargs_train["motion_network_query_fn"]
        motion_model2 = render_kwargs_train["network_motion2"]
        potentials2 = motion_query_func(pnm_points2, pnm_feature_term2, motion_model2)

        #### Construct database elements ####
        weight_rgb1, weight_features1, weight_potentials1 = SCALE_FACTORS[0]
        curr_entries1 = torch.cat([pnm_rgb_term1 * weight_rgb1, pnm_feature_term1 * weight_features1, (potentials1 - pnm_points1) * weight_potentials1], -1)

        weight_rgb2, weight_features2, weight_potentials2 = SCALE_FACTORS[1]
        curr_entries2 = torch.cat([pnm_rgb_term2 * weight_rgb2, pnm_feature_term2 * weight_features2, (potentials2 - pnm_points2) * weight_potentials2], -1)

        DATABASE1.append(curr_entries1)
        DATABASE2.append(curr_entries2)
        #####################################

      DATABASE1 = torch.cat(DATABASE1, 0)
      DATABASE2 = torch.cat(DATABASE2, 0)
      # print(DATABASE1.shape)
      # print(DATABASE2.shape)
      # exit()

      #######################################
      ####### Sampling from Database ########
      #######################################
      # NUM_Y_TO_SAMPLE = 1024
      # NUM_Y_TO_SAMPLE = 512
      NUM_Y_TO_SAMPLE = args.num_y_to_sample

      ### Select random indices 
      indices = torch.randperm(DATABASE1.shape[0])[:NUM_Y_TO_SAMPLE]  
      selected_entries = DATABASE1[indices]   

      ### Get argmin
      distances = torch.norm(selected_entries.unsqueeze(1) - DATABASE2.unsqueeze(0), p=2, dim=-1) ### (256x200k) dynamic --> (200k, 200k) dynamic
      distances_min = torch.min(distances, axis=-1)[0]
      # print(distances.shape)
      # print(distances_min.shape)
      
      # Compute loss and optimize
      optimizer_motion.zero_grad()

      ## Energy loss on the three terms
      loss = torch.mean(distances_min)

      loss.backward()
      optimizer_motion.step()

      # write logs
      if i%args.i_weights==0:
          path = os.path.join(args.ckpt_dir, args.expname, '{:06d}.tar'.format(i))
          save_dict = {
              'global_step': global_step,
              'optimizer_state_dict': optimizer_motion.state_dict()}

          save_dict["network_fn1_state_dict"] = render_kwargs_train['network_fn1'].state_dict()
          if render_kwargs_train['network_fine1'] is not None:
            save_dict["network_fine1_state_dict"] = render_kwargs_train['network_fine1'].state_dict()
          save_dict["network_motion1_state_dict"] = render_kwargs_train['network_motion1'].state_dict()

          save_dict["network_fn2_state_dict"] = render_kwargs_train['network_fn2'].state_dict()
          if render_kwargs_train['network_fine2'] is not None:
            save_dict["network_fine2_state_dict"] = render_kwargs_train['network_fine2'].state_dict()
          save_dict["network_motion2_state_dict"] = render_kwargs_train['network_motion2'].state_dict()

          torch.save(save_dict, path)
          print('Saved checkpoints at', path)
          # exit()
      
      if i%args.i_print==0:
          tb.add_scalars('motion_loss', {'train': loss.item()}, i)
          # tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}")
          print(f"[TRAIN] Iter: {i} Loss: {loss.item()}")

      global_step += 1


def config_parser():

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = configargparse.ArgumentParser()
    parser.add_argument('task', type=str, help='one out of: "train", "test", "test_with_opt", "video"')
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default="hotdog_v1", 
                        help='specify the experiment, required for "test" and "video", optional for "train"')
    parser.add_argument("--dataset", type=str, default="llff", 
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


    ### Learning rate updates
    parser.add_argument('--num_iterations', type=int, default=20000, help='Number of epochs')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument('--decay_step', type=int, default=400000, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.7]')


    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=1024*64*4, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
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
    parser.add_argument("--i_print",   type=int, default=10, 
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=1000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=1000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--ckpt_dir", type=str, default="log_motion_pair",
                        help='checkpoint directory')

    # data options
    parser.add_argument("--scene_id", type=str, default="walking_dvd",
                        help='scene identifier')
    parser.add_argument("--scene_id1", type=str, default="hotdog_single_shadowfix",
                        help='scene identifier 1 for blender')
    parser.add_argument("--scene_id2", type=str, default="hotdog_single_shadowfix_edited",
                        help='scene identifier 2 for blender')  

    parser.add_argument("--data_dir", type=str, default="/home/mikacuy/Desktop/coord-mvs/",
                        help='directory containing the scenes')

    ### Train json file --> experimenting making views sparser
    parser.add_argument("--train_jsonfile", type=str, default='transforms_train.json',
                        help='json file containing training images')

    parser.add_argument("--cimle_dir", type=str, default="dump_0718_noalign/",
                        help='dump_dir name for prior depth hypotheses')
    parser.add_argument("--num_hypothesis", type=int, default=20, 
                        help='number of cimle hypothesis')
    parser.add_argument("--space_carving_weight", type=float, default=0.007,
                        help='weight of the depth loss, values <=0 do not apply depth loss')
    parser.add_argument("--warm_start_nerf", type=int, default=0, 
                        help='number of iterations to train only vanilla nerf without additional losses.')

    ### Feature related
    parser.add_argument("--feature_dir", type=str, default="hotdog_single_shadowfix_dino_features_small/",
                        help='dump_dir name for prior depth hypotheses')
    parser.add_argument("--feature_dir1", type=str, default="hotdog_single_shadowfix_dino_features_small/",
                        help='dump_dir name for prior depth hypotheses')
    parser.add_argument("--feature_dir2", type=str, default="hotdog_single_shadowfix_edited_dino_features_small/",
                        help='dump_dir name for prior depth hypotheses') 

    parser.add_argument("--feat_dim", type=int, default=384, 
                        help='dino feature dimension')
    parser.add_argument("--feature_weight", type=float, default=0.004,
                        help='weight for feature')
    parser.add_argument("--seman_lrate", type=float, default=5e-4, 
                        help='learning rate')

    parser.add_argument('--scaleshift_lr', default= 0.00001, type=float)
    parser.add_argument('--scale_init', default= 1.0, type=float)
    parser.add_argument('--shift_init', default= 0.0, type=float)
    parser.add_argument("--freeze_ss", type=int, default=400000, 
                            help='dont update scale/shift in the last few epochs')

    ### u sampling is joint or not
    parser.add_argument('--is_joint', default= False, type=bool)

    ### Norm for space carving loss
    parser.add_argument("--norm_p", type=int, default=2, help='norm for loss')
    parser.add_argument("--space_carving_threshold", type=float, default=0.0,
                        help='threshold to not penalize the space carving loss.')
    parser.add_argument('--mask_corners', default= False, type=bool)

    parser.add_argument("--input_ch_cam", type=int, default=0,
                        help='number of channels for camera index embedding')

    parser.add_argument("--opt_ch_cam", action='store_true', default=False,
                        help='optimize camera embedding')    
    parser.add_argument('--ch_cam_lr', default= 0.0001, type=float)

    ### For loading a pair of nerf models ####
    parser.add_argument('--load_pretrained', default= False, type=bool)

    parser.add_argument("--pretrained_dir", type=str, default="/home/mikacuy/coord-mvs/google-nerf/scade_dynamics/log_blender_withdepth_dino/",
                        help='folder directory name for where the pretrained model that we want to load is')
    parser.add_argument("--pretrained_fol1", type=str, default="hotdog",
                        help='first nerf folder')
    parser.add_argument("--pretrained_fol2", type=str, default="hotdog_edited",
                        help='first nerf folder')
    ##########################################

    ### For Multi Camera setup

    # parser.add_argument(
    #     '--camera_indices',
    #     nargs='+',
    #     default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], type=list_of_ints,
    #     help='camera indices in the rig to use',
    # )  
    # parser.add_argument("--frame_idx", type=int, default=0, 
                        # help='Frame index to train the nerf model.')   
    parser.add_argument(
        '--camera_indices',
        default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], type=list_of_ints,
        help='camera indices in the rig to use',
    )     
    parser.add_argument("--frame_idx", type=list_of_ints, default=[0], 
                        help='Frame index to train the nerf model.')   
    ##################################

    parser.add_argument('--use_depth', default= False, type=bool)
    parser.add_argument("--white_bkgd", default= False, type=bool, 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    parser.add_argument("--downsample", type=int, default=8,
                        help='downsample images')

    ##### For motion module #####
    parser.add_argument("--motion_lrate", type=float, default=5e-4, 
                        help='motion module learning rate')
    parser.add_argument("--skip_views", type=int, default=20,
                        help='skip_views')
    parser.add_argument("--num_y_to_sample", type=int, default=512,
                        help='num_y_to_sample')

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

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    scene_data_dir = os.path.join(args.data_dir, args.scene_id)

    # camera_indices = np.arange(16)
    camera_indices = args.camera_indices
    frame_idx = args.frame_idx

    if args.dataset == "llff":
        images, _, _, poses, H, W, intrinsics, near, far, i_split,\
            video_poses, video_intrinsics, all_depth_hypothesis = load_llff_data_multicam_withdepth(
            scene_data_dir,
            camera_indices,
            factor=8,
            render_idx=8,
            recenter=True,
            bd_factor=4.0,
            spherify=False,
            load_imgs=True,
            frame_indices=frame_idx,
            cimle_dir=args.cimle_dir,
            num_hypothesis = args.num_hypothesis
        )
        depths = None
        valid_depths = None
    
    elif args.dataset == "scannet":
        images, _, _, poses, H, W, intrinsics, near, far, i_split,\
            video_poses, video_intrinsics, all_depth_hypothesis = load_scene_mika(scene_data_dir, camera_indices, args.cimle_dir, num_hypothesis = args.num_hypothesis,
            frame_indices = frame_idx)
        depths = None
        valid_depths = None

    elif args.dataset == "blender":
        # scene_feature_dir = os.path.join(args.data_dir, args.feature_dir)
        # images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
        #     video_poses, video_intrinsics, _, all_features, all_features_fnames  = load_scene_blender_depth_features(scene_data_dir, scene_feature_dir, downsample=args.downsample, feat_dim = args.feat_dim)

        # all_depth_hypothesis = None

        # if args.white_bkgd:
        #     images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        # else:
        #     images = images[...,:3] 

        all_images = []
        all_depths = []
        all_valid_depths = []
        all_poses = []
        all_intrinsics = []
        all_near = []
        all_far = []
        all_depth_hypothesis = []
        all_features = []
        all_features_fnames = []
        
        ### Frame 1 ###
        # Load data
        scene_data_dir = os.path.join(args.data_dir, args.scene_id1)        

        scene_feature_dir = os.path.join(args.data_dir, args.feature_dir1)
        images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
            video_poses, video_intrinsics, _, features, features_fnames  = load_scene_blender_depth_features(scene_data_dir, scene_feature_dir, downsample=args.downsample, feat_dim = args.feat_dim)

        depth_hypothesis = None

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        all_images.append(images)
        all_depths.append(depths)
        all_valid_depths.append(valid_depths)
        all_poses.append(poses)
        all_intrinsics.append(intrinsics)
        all_near.append(near)
        all_far.append(far)
        all_depth_hypothesis.append(depth_hypothesis)      
        all_features.append(features)         
        all_features_fnames.append(features_fnames)
        #################

        ### Frame 2 ###
        # Load data
        scene_data_dir = os.path.join(args.data_dir, args.scene_id2)        

        scene_feature_dir = os.path.join(args.data_dir, args.feature_dir2)
        images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
            video_poses, video_intrinsics, _, features, features_fnames  = load_scene_blender_depth_features(scene_data_dir, scene_feature_dir, downsample=args.downsample, feat_dim = args.feat_dim)

        depth_hypothesis = None

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

        all_images.append(images)
        all_depths.append(depths)
        all_valid_depths.append(valid_depths)
        all_poses.append(poses)
        all_intrinsics.append(intrinsics)
        all_near.append(near)
        all_far.append(far)
        all_depth_hypothesis.append(depth_hypothesis)    
        all_features.append(features)         
        all_features_fnames.append(features_fnames)
        #################

    all_images = np.array(all_images)
    all_depths = np.array(all_depths)
    all_valid_depths = np.array(all_valid_depths)
    all_poses = np.array(all_poses)
    all_intrinsics = np.array(all_intrinsics)
    all_near = np.array(all_near)
    all_far = np.array(all_far)
    all_depth_hypothesis = np.array(all_depth_hypothesis)
    all_features = torch.stack(all_features)

    print(all_images.shape)
    print(all_valid_depths.shape)
    print(all_poses.shape)
    print(all_intrinsics.shape)

    # print(images.shape)
    # exit()

    i_train, i_test = i_split

    # Compute boundaries of 3D space
    max_xyz = torch.full((3,), -1e6)
    min_xyz = torch.full((3,), 1e6)
    for idx_train in i_train:
        rays_o, rays_d = get_rays(H, W, torch.Tensor(intrinsics[idx_train]), torch.Tensor(poses[idx_train])) # (H, W, 3), (H, W, 3)
        points_3D = rays_o + rays_d * far # [H, W, 3]
        max_xyz = torch.max(points_3D.view(-1, 3).amax(0), max_xyz)
        min_xyz = torch.min(points_3D.view(-1, 3).amin(0), min_xyz)
    args.bb_center = (max_xyz + min_xyz) / 2.
    args.bb_scale = 2. / (max_xyz - min_xyz).max()
    print("Computed scene boundaries: min {}, max {}".format(min_xyz, max_xyz))

    scene_sample_params = {
        'precomputed_z_samples' : None,
        'near' : near,
        'far' : far,
    }

    lpips_alex = LPIPS()

    if args.task == "train":
        # train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, None, None, all_depth_hypothesis, use_depth=args.use_depth, features_fnames=all_features_fnames, features=all_features)
        train_nerf(all_images, all_depths, all_valid_depths, all_poses, all_intrinsics, i_split, args, scene_sample_params, lpips_alex, None, None, all_depth_hypothesis, use_depth=args.use_depth, features_fnames=all_features_fnames, features=all_features)
        exit()
 
    # create nerf model for testing
    _, render_kwargs_test, _, nerf_grad_vars, _, nerf_grad_names = create_nerf(args, scene_sample_params)
    for param in nerf_grad_vars:
        param.requires_grad = False

    # render test set and compute statistics
    if "test" in args.task: 
        with_test_time_optimization = False
        if args.task == "test_opt":
            with_test_time_optimization = True

        images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        intrinsics = torch.Tensor(intrinsics).to(device)
        i_test = i_test

        if args.dataset == "blender":
          depths = torch.Tensor(depths).to(device)
          valid_depths = torch.Tensor(valid_depths).bool().to(device)          
        else:
          depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2], 1)).to(device)
          valid_depths = torch.zeros((images.shape[0], images.shape[1], images.shape[2]), dtype=bool).to(device)

        print("Train split")
        print(i_train)
        print("Test split")
        print(i_test)

        mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=with_test_time_optimization, features=all_features, mode="test")
        # mean_metrics_test, images_test = render_images_with_metrics(2, i_test, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, \
        #   render_kwargs_test, with_test_time_optimization=with_test_time_optimization, features=all_features, mode="test")
        write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=with_test_time_optimization)
    elif args.task == "video":
        vposes = torch.Tensor(video_poses).to(device)
        vintrinsics = torch.Tensor(video_intrinsics).to(device)
        render_video(vposes, H, W, vintrinsics, str(0), args, render_kwargs_test, features=all_features)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    run_nerf()
