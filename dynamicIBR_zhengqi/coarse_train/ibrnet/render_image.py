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
from ibrnet.render_ray import render_rays


def render_single_image(frame_idx,
                        time_embedding,
                        time_offset, 
                        ray_sampler,
                        ray_batch,
                        model,
                        projector,
                        chunk_size,
                        N_samples,
                        args,
                        inv_uniform=False,
                        N_importance=0,
                        det=False,
                        white_bkgd=False,
                        render_stride=1,
                        featmaps=None,
                        is_train=True):
    '''
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'outputs_coarse_ref': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    '''

    all_ret = OrderedDict([('outputs_coarse_ref', OrderedDict()),
                           ('outputs_coarse_anchor', OrderedDict())])

    N_rays = ray_batch['ray_o'].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if ray_batch[k] is None:
                chunk[k] = None
            elif k in ['camera', 'anchor_camera', 'depth_range', 'src_rgbs', 'src_cameras', \
            'anchor_src_rgbs', 'anchor_src_cameras', 'static_src_rgbs', 'static_src_cameras']:
                chunk[k] = ray_batch[k]
            elif len(ray_batch[k].shape) == 3: # flow and mask
                chunk[k] = ray_batch[k][:, i:i+chunk_size, ...]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i:i+chunk_size]
            else:
                chunk[k] = None

        ret = render_rays(frame_idx=frame_idx,
                          time_embedding=time_embedding,
                          time_offset=time_offset,
                          ray_batch=chunk, 
                          model=model, 
                          featmaps=featmaps,
                          projector=projector,
                          N_samples=N_samples,
                          args=args,
                          inv_uniform=inv_uniform,
                          N_importance=N_importance,
                          raw_noise_std=0.,
                          det=det,
                          white_bkgd=white_bkgd,
                          is_train=is_train)

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret['outputs_coarse_ref']:
                all_ret['outputs_coarse_ref'][k] = []

            if is_train:
                for k in ret['outputs_coarse_anchor']:
                    all_ret['outputs_coarse_anchor'][k] = []

            if ret['outputs_fine'] is None:
                all_ret['outputs_fine'] = None
            else:
                for k in ret['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

        for k in ret['outputs_coarse_ref']:
            all_ret['outputs_coarse_ref'][k].append(ret['outputs_coarse_ref'][k].cpu())

        if is_train:
            for k in ret['outputs_coarse_anchor']:
                all_ret['outputs_coarse_anchor'][k].append(ret['outputs_coarse_anchor'][k].cpu())

        if ret['outputs_fine'] is not None:
            for k in ret['outputs_fine']:
                all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunk results and reshape
    for k in all_ret['outputs_coarse_ref']:
        if k == 'random_sigma':
            continue

        if len(all_ret['outputs_coarse_ref'][k][0].shape) == 4:
            continue

        if len(all_ret['outputs_coarse_ref'][k][0].shape) == 3:
            tmp = torch.cat(all_ret['outputs_coarse_ref'][k], dim=1).reshape((all_ret['outputs_coarse_ref'][k][0].shape[0],
                                                                          rgb_strided.shape[0],
                                                                          rgb_strided.shape[1], -1))
        else:
            tmp = torch.cat(all_ret['outputs_coarse_ref'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                          rgb_strided.shape[1], -1))
        all_ret['outputs_coarse_ref'][k] = tmp.squeeze()

    all_ret['outputs_coarse_ref']['rgb'][all_ret['outputs_coarse_ref']['mask'] == 0] = 0.

    # merge chunk results and reshape
    if is_train:
        for k in all_ret['outputs_coarse_anchor']:
            if k == 'random_sigma':
                continue

            if len(all_ret['outputs_coarse_anchor'][k][0].shape) == 4:
                continue            

            if len(all_ret['outputs_coarse_anchor'][k][0].shape) == 3:
                tmp = torch.cat(all_ret['outputs_coarse_anchor'][k], dim=1).reshape((all_ret['outputs_coarse_anchor'][k][0].shape[0],
                                                                                     rgb_strided.shape[0],
                                                                                     rgb_strided.shape[1], -1))
            else:
                tmp = torch.cat(all_ret['outputs_coarse_anchor'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                              rgb_strided.shape[1], -1))
            all_ret['outputs_coarse_anchor'][k] = tmp.squeeze()

        all_ret['outputs_coarse_anchor']['rgb'][all_ret['outputs_coarse_anchor']['mask'] == 0] = 0.

    # if all_ret['outputs_fine'] is not None:
    #     for k in all_ret['outputs_fine']:
    #         if k == 'random_sigma':
    #             continue
    #         tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape((rgb_strided.shape[0],
    #                                                                     rgb_strided.shape[1], -1))

    #         all_ret['outputs_fine'][k] = tmp.squeeze()

    return all_ret


def render_single_image_test(frame_idx,
                             time_embedding,
                             time_offset, 
                             ray_sampler,
                             ray_batch,
                             model,
                             projector,
                             chunk_size,
                             N_samples,
                             inv_uniform=False,
                             N_importance=0,
                             det=False,
                             white_bkgd=False,
                             render_stride=1,
                             featmaps=None):

    all_ret = OrderedDict([('outputs_coarse_ref', OrderedDict()),
                           ('outputs_coarse_anchor', OrderedDict())])

    N_rays = ray_batch['ray_o'].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if ray_batch[k] is None:
                chunk[k] = None
            elif k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras', 'anchor_src_rgbs', 'anchor_src_cameras']:
                chunk[k] = ray_batch[k]
            elif len(ray_batch[k].shape) == 3: # flow and mask
                chunk[k] = ray_batch[k][:, i:i+chunk_size, ...]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i:i+chunk_size]
            else:
                chunk[k] = None

        ret = render_rays(frame_idx=frame_idx,
                          time_embedding=time_embedding,
                          time_offset=time_offset,
                          ray_batch=chunk, 
                          model=model, 
                          featmaps=featmaps,
                          projector=projector,
                          N_samples=N_samples,
                          inv_uniform=inv_uniform,
                          N_importance=N_importance,
                          det=det,
                          white_bkgd=white_bkgd,
                          is_train=False)

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret['outputs_coarse_ref']:
                all_ret['outputs_coarse_ref'][k] = []

            if ret['outputs_fine'] is None:
                all_ret['outputs_fine'] = None
            else:
                for k in ret['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

        for k in ret['outputs_coarse_ref']:
            if ret['outputs_coarse_ref'][k] is None:
                continue

            all_ret['outputs_coarse_ref'][k].append(ret['outputs_coarse_ref'][k].cpu())

        if ret['outputs_fine'] is not None:
            for k in ret['outputs_fine']:
                all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunk results and reshape
    for k in all_ret['outputs_coarse_ref']:
        if k == 'random_sigma':
            continue

        if len(all_ret['outputs_coarse_ref'][k]) == 0:
            continue

        if len(all_ret['outputs_coarse_ref'][k][0].shape) == 4:
            continue

        if len(all_ret['outputs_coarse_ref'][k][0].shape) == 3:
            tmp = torch.cat(all_ret['outputs_coarse_ref'][k], dim=1).reshape((all_ret['outputs_coarse_ref'][k][0].shape[0],
                                                                              rgb_strided.shape[0],
                                                                              rgb_strided.shape[1], -1))
        else:
            tmp = torch.cat(all_ret['outputs_coarse_ref'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                              rgb_strided.shape[1], -1))
        all_ret['outputs_coarse_ref'][k] = tmp.squeeze()

    all_ret['outputs_coarse_ref']['rgb'][all_ret['outputs_coarse_ref']['mask'] == 0] = 0.

    if all_ret['outputs_fine'] is not None:
        for k in all_ret['outputs_fine']:
            if k == 'random_sigma':
                continue
            tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                        rgb_strided.shape[1], -1))

            all_ret['outputs_fine'][k] = tmp.squeeze()

    return all_ret

