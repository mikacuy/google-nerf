# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################
USE_DISTANCE = False
DETACH_SCORE = True
# OCC_WEIGHTS_MODE = 0 # 0: dy_only 1, composite_dy 2: full
USE_SOFTPLUS = True


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    '''

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)       # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)       # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i+1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)     # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o, ray_d, depth_range,
                            N_samples,
                            inv_uniform=False,
                            det=False):
    '''
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    '''
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
    if inv_uniform:
        start = 1. / near_depth     # [N_rays,]
        step = (1. / far_depth - start) / (N_samples-1)
        inv_z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples-1)
        z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand   # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o       # [N_rays, N_samples, 3]

    # from mip-nerf 360 normalized distance 
    s_vals = ((1. / z_vals) - (1. / near_depth_value)) / (1./far_depth_value - 1./near_depth_value)

    return pts, z_vals, s_vals

########################################################################################################################
# ray rendering of nerf
########################################################################################################################

def raw2outputs_vanila(raw, z_vals, mask, raw_noise_std=0., white_bkgd=False):
    '''
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''

    rgb = raw[:, :, :3]     # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]    # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    if USE_SOFTPLUS:
        sigma2alpha = lambda sigma, dists, act_fn=torch.nn.Softplus(): 1. - torch.exp(-act_fn(sigma)*dists)
    else:
        sigma2alpha = lambda sigma, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(sigma)*dists)

    # point samples are ordered with increasing depth
    # interval between samples
    if USE_DISTANCE:
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw.device)], -1)  # [N_rays, N_samples]
    else:
        dists = torch.ones_like(z_vals[...,1:])
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw.device)], -1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights = alpha * T     # [N_rays, N_samples]
    
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    # if white_bkgd:
    #     rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    mask = mask.float().sum(dim=1) > 8  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
    depth_map = torch.sum(weights * z_vals, dim=-1)     # [N_rays,]

    ret = OrderedDict([('rgb', rgb_map),
                       ('depth', depth_map),
                       ('weights', weights),                # used for importance sampling of fine samples
                       ('mask', mask),
                       ('alpha', alpha),
                       ('z_vals', z_vals)
                       ])

    return ret

def raw2outputs(raw_dy, raw_static, 
                z_vals, 
                mask_dy, mask_static, 
                raw_noise_std=0., 
                white_bkgd=False):
    '''
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''

    rgb_dy = raw_dy[:, :, :3]     # [N_rays, N_samples, 3]
    sigma_dy = raw_dy[:, :, 3]    # [N_rays, N_samples]

    rgb_static = raw_static[:, :, :3]     # [N_rays, N_samples, 3]
    sigma_static = raw_static[:, :, 3]    # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    if USE_SOFTPLUS:
        sigma2alpha = lambda sigma, dists, act_fn=torch.nn.Softplus(): 1. - torch.exp(-act_fn(sigma)*dists)
    else:
        sigma2alpha = lambda sigma, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(sigma)*dists)

    # point samples are ordered with increasing depth
    # interval between samples
    # noise = torch.randn(raw[...,3].shape).to(raw.device) * raw_noise_std

    if USE_DISTANCE:
        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw_dy.device)], -1)  # [N_rays, N_samples]
    else:
        dists = torch.ones_like(z_vals[...,1:])
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(raw_dy.device)], -1)  # [N_rays, N_samples]

    alpha_dy = sigma2alpha(sigma_dy, dists)  # [N_rays, N_samples]
    alpha_static = sigma2alpha(sigma_static, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    alpha = 1 - (1 - alpha_static) * (1 - alpha_dy)

    # alphas_sh = torch.cat([torch.ones_like(alpha[..., :1]), 1. - alpha], -1)
    # T = torch.cumprod(alphas_sh, -1)[..., :-1]

    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights_dy = alpha_dy * T     # [N_rays, N_samples]
    rgb_map_dy = torch.sum(weights_dy.unsqueeze(2) * rgb_dy, dim=1)  # [N_rays, 3]

    weights_static = alpha_static * T     # [N_rays, N_samples]
    rgb_map_static = torch.sum(weights_static.unsqueeze(2) * rgb_static, dim=1)  # [N_rays, 3]

    rgb_map = rgb_map_dy + rgb_map_static

    weights = alpha * T # (N_rays, N_samples_)

    mask = torch.bitwise_or(mask_dy.float().sum(dim=1) > 8, 
                            mask_static.float().sum(dim=1) > 8)  # should at least have 8 valid observation on the ray, otherwise don't consider its loss

    depth_map = torch.sum(weights * z_vals, dim=-1)     # [N_rays,]

    ret = OrderedDict([('rgb', rgb_map),
                       ('rgb_static', rgb_map_static),
                       ('rgb_dy', rgb_map_dy),
                       ('depth', depth_map),
                       ('weights_dy', weights_dy),
                       ('weights_st', weights_static),
                       ('alpha', alpha),
                       ('weights', weights),                # used for importance sampling of fine samples
                       ('mask', mask),
                       ('z_vals', z_vals)
                       ])

    return ret

def compute_optical_flow(outputs_coarse, raw_pts_3d_seq, src_cameras, uv_grid):
    src_cameras = src_cameras.squeeze(0)  # [n_views, 34]
    h, w = src_cameras[0][:2]

    src_intrinsics = src_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
    src_c2w = src_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
    src_w2c = torch.inverse(src_c2w)

    weights = outputs_coarse['weights'][None, ..., None]

    exp_pts_3d_seq = torch.sum(weights * raw_pts_3d_seq, dim=-2).unsqueeze(-1)

    exp_pts_3d_seq_src = torch.matmul(src_w2c[:, None, :3, :3], (exp_pts_3d_seq)) + src_w2c[:, None, :3, 3:4]
    exp_pix_time_seq_src = torch.matmul(src_intrinsics[:, None, :3, :3], exp_pts_3d_seq_src)
    exp_pix_time_seq_src = exp_pix_time_seq_src / exp_pix_time_seq_src[:, :, -1:, :]

    render_flow = exp_pix_time_seq_src[..., :2, 0] - uv_grid[None, ...]


    return render_flow

def compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, traj_basis_i):
    return torch.cat([torch.sum(raw_coeff_x * traj_basis_i, axis=-1, keepdim=True), 
               torch.sum(raw_coeff_y * traj_basis_i, axis=-1, keepdim=True), 
               torch.sum(raw_coeff_z * traj_basis_i, axis=-1, keepdim=True)], 
               dim=-1)


def compute_ref_plucker_coordinate(ray_o, ray_d):
    input_ray_d = F.normalize(ray_d, dim=-1)
    moment = torch.cross(ray_o, input_ray_d)

    return torch.cat([input_ray_d, moment], axis=-1)


def compute_src_plucker_coordinate(pts, src_cameras):
    '''
        src_cameras: c2w matrix
    '''
    src_poses = src_cameras[0, :, -16:].reshape(-1, 4, 4)
    ray_o = src_poses[:, :3, 3].unsqueeze(1).unsqueeze(1)

    if len(pts.shape) == 3:
        ray_src = pts.unsqueeze(0) - ray_o
    else:
        ray_src = pts - ray_o

    ray_src = F.normalize(ray_src, dim=-1)

    moment_src = torch.cross(ray_o.expand(-1, ray_src.shape[1], ray_src.shape[2], -1), ray_src)

    return torch.cat([ray_src, moment_src], axis=-1).permute(1, 2, 0, 3)



def render_rays(frame_idx,
                time_embedding,
                time_offset,
                ray_batch,
                model,
                featmaps,
                projector,
                N_samples,
                args,
                inv_uniform=False,
                N_importance=0,
                raw_noise_std=0.,
                det=False,
                white_bkgd=False,
                is_train=True):
    '''
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''
    OCC_WEIGHTS_MODE = args.occ_weights_mode
    ref_frame_idx, anchor_frame_idx = frame_idx[0], frame_idx[1]
    ref_time_embedding, anchor_time_embedding = time_embedding[0], time_embedding[1]
    ref_time_offset, anchor_time_offset = time_offset[0], time_offset[1]
    num_frames = int(ref_frame_idx / ref_time_embedding)

    input_ray_dir = F.normalize(ray_batch['ray_d'], dim=-1)

    ret = {'outputs_coarse': None,
           'outputs_fine': None}

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    pts_ref, z_vals, s_vals = sample_along_camera_ray(ray_o=ray_batch['ray_o'],
                                              ray_d=ray_batch['ray_d'],
                                              depth_range=ray_batch['depth_range'],
                                              N_samples=N_samples, 
                                              inv_uniform=inv_uniform, det=det)
    N_rays, N_samples = pts_ref.shape[:2]

    num_last_samples = int(round(N_samples * 0.1))

    # inference scene motion in global space
    ref_time_embedding_ = ref_time_embedding[None, None, :].repeat(N_rays, N_samples, 1)

    ref_xyzt = torch.cat([pts_ref, ref_time_embedding_], dim=-1).float().to(pts_ref.device)
    raw_coeff_xyz = model.motion_mlp(ref_xyzt)
    raw_coeff_xyz[:, -num_last_samples:, :] *= 0.0

    num_basis = model.traj_basis.shape[1]
    raw_coeff_x = raw_coeff_xyz[..., 0:num_basis]
    raw_coeff_y = raw_coeff_xyz[..., num_basis:num_basis*2]
    raw_coeff_z = raw_coeff_xyz[..., num_basis*2:num_basis*3]

    ref_traj_pts_dict = {}
    # always compute entire trajectory for target view
    for offset in [-3, -2, -1, 0, 1, 2, 3]: # HARD CODE!!!
        traj_pts_ref = compute_traj_pts(raw_coeff_x, raw_coeff_y, raw_coeff_z, 
                                        model.traj_basis[None, None, ref_frame_idx+offset, :])

        ref_traj_pts_dict[offset] = traj_pts_ref

    pts_3d_seq_ref = []
    for offset in ref_time_offset: 
        pts_3d_seq_ref.append(pts_ref + (ref_traj_pts_dict[offset] - ref_traj_pts_dict[0]))

    pts_3d_seq_ref = torch.stack(pts_3d_seq_ref, 0)
    pts_3d_static = pts_ref[None, ...].repeat(ray_batch['static_src_rgbs'].shape[1], 1, 1, 1)

    # feature query from source view with scene motions, for ref view
    rgb_feat_ref, ray_diff_ref, mask_ref = projector.compute_with_motions(pts_ref, pts_3d_seq_ref, 
                                                                          ray_batch['camera'],
                                                                          ray_batch['src_rgbs'],
                                                                          ray_batch['src_cameras'],
                                                                          featmaps=featmaps[0])  # [N_rays, N_samples, N_views, x]

    rgb_feat_static, ray_diff_static, mask_static = projector.compute_with_motions(pts_ref, pts_3d_static, 
                                                                                   ray_batch['camera'],
                                                                                   ray_batch['static_src_rgbs'],
                                                                                   ray_batch['static_src_cameras'],
                                                                                   featmaps=featmaps[2])  # [N_rays, N_samples, N_views, x]

    pixel_mask_ref = mask_ref[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations
    pixel_mask_static = mask_static[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations

    ref_time_diff = torch.from_numpy(np.array(ref_time_offset)).to(rgb_feat_ref.device) / float(num_frames)
    ref_time_diff = ref_time_diff[None, None, :, None].expand(ray_diff_ref.shape[0], ray_diff_ref.shape[1], -1, -1)

    raw_coarse_ref = model.net_coarse_dy(pts_ref,
                                         rgb_feat_ref, 
                                         input_ray_dir,
                                         ray_diff_ref, 
                                         ref_time_diff,
                                         mask_ref, 
                                         ref_time_embedding_)   # [N_rays, N_samples, 4]

    ref_rays_coords = compute_ref_plucker_coordinate(ray_batch['ray_o'], ray_batch['ray_d'])
    src_rays_coords = compute_src_plucker_coordinate(pts_ref, ray_batch['static_src_cameras'])

    raw_coarse_static = model.net_coarse_st(pts_ref,
                                            ref_rays_coords,
                                            src_rays_coords,
                                            rgb_feat_static, 
                                            input_ray_dir,
                                            ray_diff_static, 
                                            mask_static)   # [N_rays, N_samples, 4]

    outputs_coarse_ref = raw2outputs(raw_coarse_ref, raw_coarse_static,
                                     z_vals, 
                                     pixel_mask_ref, pixel_mask_static)

    outputs_coarse_ref_dy = raw2outputs_vanila(raw_coarse_ref,
                                               z_vals, 
                                               pixel_mask_ref)

    render_flows = compute_optical_flow(outputs_coarse_ref, 
                                        pts_3d_seq_ref, 
                                        ray_batch['src_cameras'], 
                                        ray_batch['uv_grid'])
    outputs_coarse_ref['render_flows'] = render_flows
    outputs_coarse_ref['s_vals'] = s_vals
    
    sf_fields = torch.norm(ref_traj_pts_dict[2] - ref_traj_pts_dict[0], dim=-1)
    weights = outputs_coarse_ref['weights']

    outputs_coarse_st = raw2outputs_vanila(raw_coarse_static,
                                           z_vals, 
                                           pixel_mask_static)

    exp_sf_p1 = torch.sum(outputs_coarse_ref['weights'][..., None] * (ref_traj_pts_dict[2] - ref_traj_pts_dict[0]), dim=-2)
    exp_sf_m1 = torch.sum(outputs_coarse_ref['weights'][..., None] * (ref_traj_pts_dict[-2] - ref_traj_pts_dict[0]), dim=-2)
    outputs_coarse_ref['exp_sf'] = torch.max(exp_sf_p1, exp_sf_m1)

    # ============================= for trajectory consistency, and temporal consistency
    if is_train:
        # compute scene flow
        sf_seq = []
        for offset in [-2, -1, 0, 1, 2, 3]: 
            sf = ref_traj_pts_dict[offset] - ref_traj_pts_dict[offset - 1] 
            sf_seq.append(sf)
        sf_seq = torch.stack(sf_seq, 0)

        # THIS PART IS FOR CYCLE CONSISTENCY
        pts_anchor = pts_ref + (ref_traj_pts_dict[anchor_frame_idx - ref_frame_idx] - ref_traj_pts_dict[0])
        anchor_time_embedding_ = anchor_time_embedding[None, None, :].repeat(N_rays, N_samples, 1).float()

        xyzt_anchor = torch.cat([pts_anchor, anchor_time_embedding_], dim=-1).float().to(pts_ref.device)

        raw_coeff_xyz_anchor = model.motion_mlp(xyzt_anchor)
        raw_coeff_xyz_anchor[:, -num_last_samples:, :] *= 0.0

        raw_coeff_x_anchor = raw_coeff_xyz_anchor[..., 0:num_basis]
        raw_coeff_y_anchor = raw_coeff_xyz_anchor[..., num_basis:num_basis*2]
        raw_coeff_z_anchor = raw_coeff_xyz_anchor[..., num_basis*2:num_basis*3]

        traj_pts_anchor_0 = compute_traj_pts(raw_coeff_x_anchor, raw_coeff_y_anchor, raw_coeff_z_anchor, 
                                             model.traj_basis[None, None, anchor_frame_idx, :])

        pts_3d_seq_anchor = []
        pts_traj_ref = []
        pts_traj_anchor = []

        # offset from nearby anchor image
        for offset in anchor_time_offset: 
            ref_offset = (anchor_frame_idx + offset - ref_frame_idx)

            traj_pts_anchor = compute_traj_pts(raw_coeff_x_anchor, raw_coeff_y_anchor, raw_coeff_z_anchor, 
                                               model.traj_basis[None, None, anchor_frame_idx + offset, :])

            temp_pts = pts_anchor + (traj_pts_anchor - traj_pts_anchor_0)
            pts_3d_seq_anchor.append(temp_pts)
            
            if ref_offset not in ref_traj_pts_dict:
                continue

            # print("ref_offset %d %d"%(ref_offset, ref_frame_idx + ref_offset), 'anchor_src %d %d'%(offset, anchor_frame_idx + offset))

            pts_traj_anchor.append(temp_pts)
            pts_traj_ref.append(pts_ref + ref_traj_pts_dict[ref_offset] - ref_traj_pts_dict[0])

        pts_traj_ref = torch.stack(pts_traj_ref, 0)
        pts_traj_anchor = torch.stack(pts_traj_anchor, 0)
        pts_3d_seq_anchor = torch.stack(pts_3d_seq_anchor, 0)

        # feature query from double-source view with scene motions, for randomly selected nearby view
        rgb_feat_anchor, ray_diff_anchor, mask_anchor = projector.compute_with_motions(pts_ref, pts_3d_seq_anchor, 
                                                                                       ray_batch['camera'], 
                                                                                       ray_batch['anchor_src_rgbs'],
                                                                                       ray_batch['anchor_src_cameras'],
                                                                                       featmaps=featmaps[1])  # [N_rays, N_samples, N_views, x]
        
        anchor_time_diff = torch.from_numpy(np.array(anchor_time_offset)).to(rgb_feat_anchor.device) / float(num_frames)
        anchor_time_diff = anchor_time_diff[None, None, :, None].expand(ray_diff_anchor.shape[0], ray_diff_anchor.shape[1], -1, -1)

        pixel_mask_anchor = mask_anchor[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations
        raw_coarse_anchor = model.net_coarse_dy(pts_anchor,
                                                rgb_feat_anchor, 
                                                input_ray_dir,
                                                ray_diff_anchor, 
                                                anchor_time_diff,
                                                mask_anchor, 
                                                anchor_time_embedding_)   # [N_rays, N_samples, 4]

        outputs_coarse_anchor = raw2outputs(raw_coarse_anchor, raw_coarse_static,
                                            z_vals, 
                                            pixel_mask_anchor, pixel_mask_static)


        outputs_coarse_anchor_dy = raw2outputs_vanila(raw_coarse_anchor,
                                                      z_vals, 
                                                      pixel_mask_anchor)

        occ_score_dy = outputs_coarse_ref_dy['weights'] - outputs_coarse_anchor_dy['weights']
        occ_score_dy = occ_score_dy.detach()
        occ_weights_dy = 1. - torch.abs(occ_score_dy)
        occ_weight_dy_map = 1. - torch.abs(torch.sum(occ_score_dy , dim=1))

        # compute disocculusion weights for full rendering
        if OCC_WEIGHTS_MODE == 0: # mix-mode
            time_diff = np.abs(ref_frame_idx - anchor_frame_idx)
            # compute disocculusion weights for full rendering
            if time_diff > 1: # composite-dy
                occ_score = outputs_coarse_ref['weights_dy'] - outputs_coarse_anchor['weights_dy']
            else: # full
                occ_score = outputs_coarse_ref['weights'] - outputs_coarse_anchor['weights']

        elif OCC_WEIGHTS_MODE == 1: # composite-dy
            occ_score = outputs_coarse_ref['weights_dy'] - outputs_coarse_anchor['weights_dy']
        elif OCC_WEIGHTS_MODE == 2: # full
            occ_score = outputs_coarse_ref['weights'] - outputs_coarse_anchor['weights']
        elif OCC_WEIGHTS_MODE == 3:
            time_diff = np.abs(ref_frame_idx - anchor_frame_idx)
            # compute disocculusion weights for full rendering
            if time_diff > 2: # composite-dy
                occ_score = outputs_coarse_ref['weights_dy'] - outputs_coarse_anchor['weights_dy']
            else: # full
                occ_score = outputs_coarse_ref['weights'] - outputs_coarse_anchor['weights']
        else:
            raise ValueError


        occ_score = occ_score.detach()

        occ_weights = 1. - torch.abs(occ_score)
        occ_weight_map = 1. - torch.abs(torch.sum(occ_score , dim=1))

        outputs_coarse_anchor['occ_weights'] = occ_weights
        outputs_coarse_anchor['occ_weight_map'] = occ_weight_map

        outputs_coarse_anchor['pts_traj_ref'] = pts_traj_ref
        outputs_coarse_anchor['pts_traj_anchor'] = pts_traj_anchor
        outputs_coarse_anchor['sf_seq'] = sf_seq


        outputs_coarse_anchor_dy['occ_weights'] = occ_weights_dy
        outputs_coarse_anchor_dy['occ_weight_map'] = occ_weight_dy_map

        ret['outputs_coarse_anchor'] = outputs_coarse_anchor
        ret['outputs_coarse_anchor_dy'] = outputs_coarse_anchor_dy

    # ======================== end for consistency
    ret['outputs_coarse_ref'] = outputs_coarse_ref
    ret['outputs_coarse_ref_dy'] = outputs_coarse_ref_dy
    ret['outputs_coarse_st'] = outputs_coarse_st
    return ret
