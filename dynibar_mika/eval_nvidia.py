"""Evaluation script for the Nvidia Benchmark."""

import collections
import math
import os
import time
from config import config_parser
import cv2
from ibrnet.data_loaders.llff_data_utils import batch_parse_llff_poses
from ibrnet.data_loaders.llff_data_utils import load_llff_data
from ibrnet.model import DynibarFF
from ibrnet.projection import Projector
from ibrnet.render_image import render_single_image_nvi
from ibrnet.sample_ray import RaySamplerSingleImage
import imageio
import models
import numpy as np
import skimage.metrics
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import *

#### Flow Visualization ###
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


############################


class DynamicVideoDataset(Dataset):
  """This class loads data from Nvidia benchmarks, including camera scene and image information from source views."""

  def __init__(self, render_idx, args, scenes, **kwargs):
    self.folder_path = args.folder_path
    self.render_idx = render_idx
    self.mask_static = args.mask_static

    print('loading {} for rendering'.format(scenes))
    assert len(scenes) == 1

    scene = scenes[0]
    self.scene_path = os.path.join(
        self.folder_path, scene, 'dense'
    )
    _, poses, bds, _, i_test, rgb_files, _ = load_llff_data(
        self.scene_path,
        height=288,
        num_avg_imgs=12,
        render_idx=self.render_idx,
        load_imgs=False,
    )
    near_depth = np.min(bds)
    # Adding 15 to ensure we cover far scene contents
    far_depth = np.max(bds) + 15.0
    self.num_frames = len(rgb_files)

    intrinsics, c2w_mats = batch_parse_llff_poses(poses)
    h, w = poses[0][:2, -1]
    render_intrinsics, render_c2w_mats = (
        intrinsics,
        c2w_mats,
    )

    self.train_intrinsics = intrinsics
    self.train_poses = c2w_mats
    self.train_rgb_files = rgb_files 
    self.render_intrinsics = render_intrinsics

    self.render_poses = render_c2w_mats 
    self.render_depth_range = [[near_depth, far_depth]] * self.num_frames
    self.h = [int(h)] * self.num_frames
    self.w = [int(w)] * self.num_frames

  def __len__(self):
    return 12 # number of viewpoints

  def __getitem__(self, idx):
    render_pose = self.render_poses[idx]
    intrinsics = self.render_intrinsics[idx]
    depth_range = self.render_depth_range[idx]

    train_rgb_files = self.train_rgb_files
    train_poses = self.train_poses
    train_intrinsics = self.train_intrinsics

    h, w = self.h[idx], self.w[idx]
    camera = np.concatenate(
        ([h, w], intrinsics.flatten(), render_pose.flatten())
    ).astype(np.float32)

    gt_img_path = os.path.join(
        self.scene_path,
        'mv_images',
        '%05d' % self.render_idx,
        'cam%02d.jpg' % (idx + 1),
    )

    nearest_pose_ids = np.sort(
        [self.render_idx + offset for offset in [1, 2, 3, 0, -1, -2, -3]]
    )
    # 12 is number of viewpoints we sample from input cameras
    num_imgs_per_cycle = 12

    # Get camera viewpoint that is closet to target view using index for benchmark
    # Since benchamrk has fixed viewpoint in a round-robin manner
    static_pose_ids = np.array(list(range(0, train_poses.shape[0])))
    static_id_dict = collections.defaultdict(list)
    for static_pose_id in static_pose_ids:
      # do not include image with the same viewpoint
      if (
          static_pose_id % num_imgs_per_cycle
          == self.render_idx % num_imgs_per_cycle
      ):
        continue

      static_id_dict[static_pose_id % num_imgs_per_cycle].append(static_pose_id)

    static_pose_ids = []
    for key in static_id_dict:
      min_idx = np.argmin(
          np.abs(np.array(static_id_dict[key]) - self.render_idx)
      )
      static_pose_ids.append(static_id_dict[key][min_idx])

    static_pose_ids = np.sort(static_pose_ids)

    src_rgbs = []
    src_cameras = []
    for src_idx in nearest_pose_ids:
      src_rgb = (
          imageio.v2.imread(train_rgb_files[src_idx]).astype(np.float32) / 255.0
      )
      train_pose = train_poses[src_idx]
      train_intrinsics_ = train_intrinsics[src_idx]
      src_rgbs.append(src_rgb)
      img_size = src_rgb.shape[:2]
      src_camera = np.concatenate(
          (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
      ).astype(np.float32)

      src_cameras.append(src_camera)

    src_rgbs = np.stack(src_rgbs, axis=0)
    src_cameras = np.stack(src_cameras, axis=0)

    static_src_rgbs = []
    static_src_cameras = []
    static_src_masks = []

    # load src rgb for static view
    for st_near_id in static_pose_ids:
      src_rgb = (
          imageio.v2.imread(train_rgb_files[st_near_id]).astype(np.float32)
          / 255.0
      )
      train_pose = train_poses[st_near_id]
      train_intrinsics_ = train_intrinsics[st_near_id]

      static_src_rgbs.append(src_rgb)

      # load coarse mask
      if self.mask_static and 3 <= st_near_id < self.num_frames - 3:
        st_mask_path = os.path.join(
            '/'.join(train_rgb_files[st_near_id].split('/')[:-2]),
            'coarse_masks',
            '%05d.png' % st_near_id,
        )
        st_mask = imageio.v2.imread(st_mask_path).astype(np.float32) / 255.0
        st_mask = cv2.resize(
            st_mask,
            (src_rgb.shape[1], src_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
      else:
        st_mask = np.ones_like(src_rgb[..., 0])

      static_src_masks.append(st_mask)

      img_size = src_rgb.shape[:2]
      src_camera = np.concatenate(
          (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
      ).astype(np.float32)

      static_src_cameras.append(src_camera)

    static_src_rgbs = np.stack(static_src_rgbs, axis=0)
    static_src_cameras = np.stack(static_src_cameras, axis=0)
    static_src_masks = np.stack(static_src_masks, axis=0)

    depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

    return {
        'camera': torch.from_numpy(camera),
        'rgb_path': gt_img_path,
        'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
        'src_cameras': torch.from_numpy(src_cameras).float(),
        'static_src_rgbs': torch.from_numpy(static_src_rgbs[..., :3]).float(),
        'static_src_cameras': torch.from_numpy(static_src_cameras).float(),
        'static_src_masks': torch.from_numpy(static_src_masks).float(),
        'depth_range': depth_range,
        'ref_time': float(self.render_idx / float(self.num_frames)),
        'id': self.render_idx,
        'nearest_pose_ids': nearest_pose_ids,
    }


def calculate_psnr(img1, img2, mask):
  """Compute PSNR between two images.

  Args:
    img1: image 1
    img2: image 2
    mask: mask indicating which region is valid.

  Returns:
    PSNR: PSNR error
  """

  # img1 and img2 have range [0, 1]
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  mask = mask.astype(np.float64)

  num_valid = np.sum(mask) + 1e-8

  mse = np.sum((img1 - img2) ** 2 * mask) / num_valid

  if mse == 0:
    return 0  # float('inf')

  return 10 * math.log10(1.0 / mse)


def calculate_ssim(img1, img2, mask):
  """Compute SSIM between two images.

  Args:
    img1: image 1
    img2: image 2
    mask: mask indicating which region is valid.

  Returns:
    PSNR: PSNR error
  """
  if img1.shape != img2.shape:
    raise ValueError('Input images must have the same dimensions.')

  _, ssim_map = skimage.metrics.structural_similarity(
      img1, img2, multichannel=True, full=True
  )
  num_valid = np.sum(mask) + 1e-8

  return np.sum(ssim_map * mask) / num_valid


def im2tensor(image, cent=1.0, factor=1.0 / 2.0):
  """Convert image to Pytorch tensor.

  Args:
    image: input image
    cent: shift
    factor: scale

  Returns:
    Pytorch tensor
  """
  return torch.Tensor(
      (image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
  )


if __name__ == '__main__':
  parser = config_parser()
  args = parser.parse_args()
  args.distributed = False
  # Construct a dataset to get number of frames for evaluation
  test_dataset = DynamicVideoDataset(0, args, scenes=args.eval_scenes)
  args.num_frames = test_dataset.num_frames
  print('args.num_frames ', args.num_frames)
  # Create ibrnet model
  model = DynibarFF(args, load_scheduler=False, load_opt=False)
  eval_dataset_name = args.eval_dataset
  extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
  print('saving results to {}...'.format(extra_out_dir))
  os.makedirs(extra_out_dir, exist_ok=True)

  projector = Projector(device='cuda:0')

  assert len(args.eval_scenes) == 1, 'only accept single scene'
  scene_name = args.eval_scenes[0]
  out_scene_dir = os.path.join(extra_out_dir, 'renderings')
  out_flow_dir = os.path.join(extra_out_dir, 'pred_flow_flipped')
  out_basis_dir = os.path.join(extra_out_dir, 'basis')
  print('saving results to {}'.format(out_scene_dir))
  os.makedirs(out_scene_dir, exist_ok=True)
  os.makedirs(out_flow_dir, exist_ok=True)
  os.makedirs(out_basis_dir, exist_ok=True)


  lpips_model = models.PerceptualLoss(
      model='net-lin', net='alex', use_gpu=True, version=0.1
  )

  psnr_list = []
  ssim_list = []
  lpips_list = []

  dy_psnr_list = []
  dy_ssim_list = []
  dy_lpips_list = []

  st_psnr_list = []
  st_ssim_list = []
  st_lpips_list = []

  for img_i in range(3, args.num_frames - 3): 
    test_dataset = DynamicVideoDataset(img_i, args, scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=12, shuffle=False
    )
    total_num = len(test_loader)
    out_frames = []

    if img_i > 12:
      exit()

    for i, data in enumerate(test_loader):
      print('img_i ', img_i, i)

      if img_i % 12 == i:
        continue

      # idx = int(data['id'].item())
      start = time.time()

      ref_time_embedding = data['ref_time'].cuda()
      ref_frame_idx = int(data['id'].item())
      ref_time_offset = [
          int(near_idx - ref_frame_idx)
          for near_idx in data['nearest_pose_ids'].squeeze().tolist()
      ]

      model.switch_to_eval()
      with torch.no_grad():
        ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
        ray_batch = ray_sampler.get_all()

        cb_featmaps_1, cb_featmaps_2 = model.feature_net(
            ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
        )
        ref_featmaps = cb_featmaps_1

        static_src_rgbs = (
            ray_batch['static_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
        )
        _, static_featmaps = model.feature_net(static_src_rgbs)

        cb_featmaps_1_fine, _ = model.feature_net_fine(
            ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
        )
        ref_featmaps_fine = cb_featmaps_1_fine

        if args.mask_static:
          static_src_rgbs_ = (
              static_src_rgbs
              * ray_batch['static_src_masks'].squeeze(0)[:, None, ...]
          )
        else:
          static_src_rgbs_ = static_src_rgbs

        _, static_featmaps_fine = model.feature_net_fine(static_src_rgbs_)

        ret = render_single_image_nvi(
            frame_idx=(ref_frame_idx, None),
            time_embedding=(ref_time_embedding, None),
            time_offset=(ref_time_offset, None),
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            det=True,
            N_samples=args.N_samples,
            args=args,
            inv_uniform=args.inv_uniform,
            N_importance=args.N_importance,
            white_bkgd=args.white_bkgd,
            coarse_featmaps=(ref_featmaps, None, static_featmaps),
            fine_featmaps=(ref_featmaps_fine, None, static_featmaps_fine),
            is_train=False,
        )

        # print(ret['outputs_coarse_ref']['outputs_coarse_rendered_flow'].shape)
        # exit()

      # rendered_flow_all = ret['outputs_coarse_ref']['outputs_coarse_rendered_flow'].detach().cpu().numpy()
      rendered_flow_all = ret['outputs_coarse_ref']['outputs_coarse_rendered_flow']
      rendered_pts_all = ret['outputs_coarse_ref']['outputs_coarse_rendered_pts']

      rendered_phis = ret['outputs_coarse_ref']['rendered_phis']

      ### basis is the same -- same vector only time_idx dependent
      trajectory_basis = ret['outputs_coarse_ref']['trajectory_basis'][0].detach().cpu().numpy()
      np.savetxt(os.path.join(out_basis_dir, str(img_i) + "_" + str(i) + "_hbasis.txt"), trajectory_basis)

      print(rendered_phis.shape)
      # print(rendered_pts_all.shape)
      # exit()
      basis_projected, _ = projector.compute_projections(rendered_phis, data["camera"].repeat(rendered_phis.shape[0], 1))
      print(basis_projected.shape)
      # exit()

      for b in range(basis_projected.shape[0]):
        print(basis_projected[b])
        print(basis_projected[b].shape)
        
        # flow_color = flow_vis.flow_to_color(optical_flow[neighbor], convert_to_bgr=False)
        flow_color = flow_to_color(basis_projected[b].detach().cpu().numpy(), clip_flow=None, convert_to_bgr=False)
        cv2.imwrite(os.path.join(out_basis_dir, str(img_i) + "_" + str(i) + "_phi" + str(b) + ".jpg"), flow_color)
      
      # exit()

      # print(trajectory_basis)
      # print(trajectory_basis.shape)

      # exit()

      fine_pred_rgb = ret['outputs_fine_ref']['rgb'].detach().cpu().numpy()
      fine_pred_depth = ret['outputs_fine_ref']['depth'].detach().cpu().numpy()

      # print(rendered_flow_all[0])
      print(rendered_flow_all[0].shape)
      print(rendered_pts_all[0].shape)
      # exit()

      # print()
      # print(fine_pred_rgb)
      # print(data["camera"])
      print(data["camera"].shape)
      print(data["camera"].repeat(rendered_pts_all.shape[0], 1).shape)

      pts_flow_projected, _ = projector.compute_projections(rendered_pts_all, data["camera"].repeat(rendered_pts_all.shape[0], 1))

      ### middle frame is the reference
      optical_flow = pts_flow_projected - pts_flow_projected[3]

      for neighbor in range(optical_flow.shape[0]):
        print(optical_flow[neighbor])
        print(optical_flow[neighbor].shape)
        
        # flow_color = flow_vis.flow_to_color(optical_flow[neighbor], convert_to_bgr=False)
        flow_color = flow_to_color(optical_flow[neighbor].detach().cpu().numpy(), clip_flow=None, convert_to_bgr=False)
        print(flow_color.shape)

        ### middle frame with idx 3 is the reference
        neighbor_idx = neighbor -3

        cv2.imwrite(os.path.join(out_flow_dir, str(img_i) + "_" + str(i) + "_flow" + str(neighbor_idx) + ".jpg"), flow_color)
      
      # exit()
      # print(optical_flow[3])
      # print(optical_flow[2])
      # print(optical_flow.shape)
      # exit()

      valid_mask = np.float32(
          np.sum(fine_pred_rgb, axis=-1, keepdims=True) > 1e-3
      )
      valid_mask = np.tile(valid_mask, (1, 1, 3))
      gt_img = cv2.imread(data['rgb_path'][0])[:, :, ::-1]
      gt_img = cv2.resize(
          gt_img,
          dsize=(fine_pred_rgb.shape[1], fine_pred_rgb.shape[0]),
          interpolation=cv2.INTER_AREA,
      )
      gt_img = np.float32(gt_img) / 255

      gt_img = gt_img * valid_mask
      fine_pred_rgb = fine_pred_rgb * valid_mask

      dynamic_mask = valid_mask
      ssim = calculate_ssim(gt_img, fine_pred_rgb, dynamic_mask)
      psnr = calculate_psnr(gt_img, fine_pred_rgb, dynamic_mask)

      gt_img_0 = im2tensor(gt_img).cuda()
      fine_pred_rgb_0 = im2tensor(fine_pred_rgb).cuda()
      dynamic_mask_0 = torch.Tensor(
          dynamic_mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
      )

      lpips = lpips_model.forward(
          gt_img_0, fine_pred_rgb_0, dynamic_mask_0
      ).item()
      print(psnr, ssim, lpips)
      psnr_list.append(psnr)
      ssim_list.append(ssim)
      lpips_list.append(lpips)

      dynamic_mask_path = os.path.join(
          test_dataset.scene_path,
          'mv_masks',
          '%05d' % img_i,
          'cam%02d.png' % (i + 1),
      )

      dynamic_mask = np.float32(cv2.imread(dynamic_mask_path) > 1e-3)  # /255.
      dynamic_mask = cv2.resize(
          dynamic_mask,
          dsize=(gt_img.shape[1], gt_img.shape[0]),
          interpolation=cv2.INTER_NEAREST,
      )

      dynamic_mask_0 = torch.Tensor(
          dynamic_mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
      )
      dynamic_ssim = calculate_ssim(gt_img, fine_pred_rgb, dynamic_mask)
      dynamic_psnr = calculate_psnr(gt_img, fine_pred_rgb, dynamic_mask)
      dynamic_lpips = lpips_model.forward(
          gt_img_0, fine_pred_rgb_0, dynamic_mask_0
      ).item()
      print(dynamic_psnr, dynamic_ssim, dynamic_lpips)

      dy_psnr_list.append(dynamic_psnr)
      dy_ssim_list.append(dynamic_ssim)
      dy_lpips_list.append(dynamic_lpips)

      static_mask = 1 - dynamic_mask
      static_mask_0 = torch.Tensor(
          static_mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
      )
      static_ssim = calculate_ssim(gt_img, fine_pred_rgb, static_mask)
      static_psnr = calculate_psnr(gt_img, fine_pred_rgb, static_mask)
      static_lpips = lpips_model.forward(
          gt_img_0, fine_pred_rgb_0, static_mask_0
      ).item()
      print(static_psnr, static_ssim, static_lpips)

      st_psnr_list.append(static_psnr)
      st_ssim_list.append(static_ssim)
      st_lpips_list.append(static_lpips)

      cv2.imwrite(os.path.join(out_scene_dir, str(img_i) + "_" + str(i) + "_0rgb" + ".jpg"), cv2.cvtColor(to8b(fine_pred_rgb), cv2.COLOR_RGB2BGR))
      cv2.imwrite(os.path.join(out_scene_dir, str(img_i) + "_" + str(i) + "_1gt" + ".jpg"), cv2.cvtColor(to8b(gt_img), cv2.COLOR_RGB2BGR))
      cv2.imwrite(os.path.join(out_scene_dir, str(img_i) + "_" + str(i) + "_2mask" + ".jpg"), cv2.cvtColor(to8b(dynamic_mask), cv2.COLOR_RGB2BGR))

    print('MOVING PSNR ', np.mean(np.array(psnr_list)))
    print('MOVING SSIM ', np.mean(np.array(ssim_list)))
    print('MOVING LPIPS ', np.mean(np.array(lpips_list)))

    print('MOVING DYNAMIC PSNR ', np.mean(np.array(dy_psnr_list)))
    print('MOVING DYNAMIC SSIM ', np.mean(np.array(dy_ssim_list)))
    print('MOVING DYNAMIC LPIPS ', np.mean(np.array(dy_lpips_list)))

    print('MOVING Static PSNR ', np.mean(np.array(st_psnr_list)))
    print('MOVING Static SSIM ', np.mean(np.array(st_ssim_list)))
    print('MOVING Static LPIPS ', np.mean(np.array(st_lpips_list)))

  print('AVG PSNR ', np.mean(np.array(psnr_list)))
  print('AVG SSIM ', np.mean(np.array(ssim_list)))
  print('AVG LPIPS ', np.mean(np.array(lpips_list)))

  print('AVG DYNAMIC PSNR ', np.mean(np.array(dy_psnr_list)))
  print('AVG DYNAMIC SSIM ', np.mean(np.array(dy_ssim_list)))
  print('AVG DYNAMIC LPIPS ', np.mean(np.array(dy_lpips_list)))

  print('AVG Static PSNR ', np.mean(np.array(st_psnr_list)))
  print('AVG Static SSIM ', np.mean(np.array(st_ssim_list)))
  print('AVG Static LPIPS ', np.mean(np.array(st_lpips_list)))