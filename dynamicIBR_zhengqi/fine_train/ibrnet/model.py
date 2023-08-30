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
import os
from ibrnet.mlp_network import IBRNetStatic, IBRNetDynamic, MotionMLP
from ibrnet.feature_network import ResUNet, ResNet
import numpy as np

def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################
def init_dct_basis(num_basis, num_frames):
    T = num_frames
    K = num_basis
    # for each t
    dct_basis = torch.zeros([T, K])

    for t in range(T):
        for k in range(1, K+1):
            dct_basis[t, k - 1] = np.sqrt(2. / T) * np.cos(np.pi / (2. * T) * (2 * t + 1) * k)

    return dct_basis

class IBRNetModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.local_rank))
        # create coarse IBRNet
        self.net_coarse_st = IBRNetStatic(args,
                                          in_feat_ch=self.args.coarse_feat_dim,
                                          n_samples=self.args.N_samples).to(self.device)

        self.net_coarse_dy = IBRNetDynamic(args,
                                           in_feat_ch=self.args.coarse_feat_dim,
                                           n_samples=self.args.N_samples).to(self.device)

        # create fine IBRNet
        self.net_fine_st = IBRNetStatic(args,
                                        in_feat_ch=self.args.fine_feat_dim,
                                        n_samples=self.args.N_samples+self.args.N_importance).to(self.device)
        self.net_fine_dy = IBRNetDynamic(args,
                                         in_feat_ch=self.args.fine_feat_dim,
                                         n_samples=self.args.N_samples+self.args.N_importance).to(self.device)

        # create coarse feature extraction network
        self.feature_net = ResNet(coarse_out_ch=self.args.coarse_feat_dim,
                                  fine_out_ch=self.args.fine_feat_dim,
                                  coarse_only=False).to(self.device)

        # create fine feature extraction network for fine stage
        self.feature_net_fine = ResNet(coarse_out_ch=self.args.coarse_feat_dim,
                                       fine_out_ch=self.args.fine_feat_dim,
                                       coarse_only=False).to(self.device)

        # Scene Flow MLPMotionMLP
        self.motion_mlp = MotionMLP(num_basis=args.num_basis).float().to(self.device)
        self.motion_mlp_fine = MotionMLP(num_basis=args.num_basis).float().to(self.device)

        # basis
        dct_basis = init_dct_basis(args.num_basis, args.num_frames)
        self.traj_basis = torch.nn.parameter.Parameter(dct_basis).float().to(self.device).detach().requires_grad_(True)
        self.traj_basis_fine = torch.nn.parameter.Parameter(dct_basis).float().to(self.device).detach().requires_grad_(True)

        print("lrate_decay_steps ", args.lrate_decay_steps, ' lrate_decay_factor ', args.lrate_decay_factor)

        self.load_coarse_from_ckpt(args.coarse_dir,
                                   load_opt=False,
                                   load_scheduler=False)

        out_folder = os.path.join(args.rootdir, 'out', args.expname)


        self.optimizer = torch.optim.Adam([
            {'params': self.net_fine_st.parameters(), 'lr': args.lrate_mlp * args.lr_multipler},
            {'params': self.net_fine_dy.parameters(), 'lr': args.lrate_mlp},
            {'params': self.feature_net_fine.parameters(), 'lr': args.lrate_feature},
            {'params': self.motion_mlp_fine.parameters(), 'lr': args.lrate_mlp},
            {'params': self.traj_basis_fine, 'lr': args.lrate_mlp * 0.2},
            ])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        self.start_step = self.load_fine_from_ckpt(out_folder,
                                                   load_opt=True,
                                                   load_scheduler=True)

        # print("start_step ", start_step)
        # sys.exit()

        # turn off gradient for coarse networks
        self.net_coarse_st.requires_grad_(False).eval()
        self.net_coarse_dy.requires_grad_(False).eval()
        self.feature_net.requires_grad_(False).eval()
        self.motion_mlp.requires_grad_(False).eval()

        device_ids = list(range(torch.cuda.device_count()))
        # parallel for coarse network
        self.net_coarse_st = torch.nn.DataParallel(self.net_coarse_st, device_ids=device_ids)
        self.net_coarse_dy = torch.nn.DataParallel(self.net_coarse_dy, device_ids=device_ids)
        self.feature_net = torch.nn.DataParallel(self.feature_net, device_ids=device_ids)
        self.motion_mlp = torch.nn.DataParallel(self.motion_mlp, device_ids=device_ids)
        # parallel for fine network
        self.net_fine_st = torch.nn.DataParallel(self.net_fine_st, device_ids=device_ids)
        self.net_fine_dy = torch.nn.DataParallel(self.net_fine_dy, device_ids=device_ids)
        self.feature_net_fine = torch.nn.DataParallel(self.feature_net_fine, device_ids=device_ids)
        self.motion_mlp_fine = torch.nn.DataParallel(self.motion_mlp_fine, device_ids=device_ids)

    def switch_to_eval(self):
        self.net_fine_st.eval()
        self.net_fine_dy.eval()

        self.feature_net_fine.eval()
        self.motion_mlp_fine.eval()

    def switch_to_train(self):
        self.net_fine_st.train()
        self.net_fine_dy.train()

        self.feature_net_fine.train()
        self.motion_mlp_fine.train()

    def save_model(self, filename, global_step):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'net_fine_st': de_parallel(self.net_fine_st).state_dict(),
                   'net_fine_dy': de_parallel(self.net_fine_dy).state_dict(),
                   'feature_net_fine': de_parallel(self.feature_net_fine).state_dict(),
                   'motion_mlp_fine': de_parallel(self.motion_mlp_fine).state_dict(),
                   'traj_basis_fine': self.traj_basis_fine,
                   "global_step": int(global_step),
                   }

        torch.save(to_save, filename)

    def load_coarse_model(self, filename, load_opt=True, load_scheduler=True):

        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        self.net_coarse_st.load_state_dict(to_load['net_coarse_st'])
        self.net_coarse_dy.load_state_dict(to_load['net_coarse_dy'])

        self.feature_net.load_state_dict(to_load['feature_net'])

        self.motion_mlp.load_state_dict(to_load['motion_mlp'])
        self.traj_basis = (to_load['traj_basis'])

        return to_load['global_step']

    def load_coarse_model2(self, filename, load_opt=True, load_scheduler=True):

        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        self.net_fine_st.load_state_dict(to_load['net_coarse_st'])
        self.net_fine_dy.load_state_dict(to_load['net_coarse_dy'])

        self.feature_net_fine.load_state_dict(to_load['feature_net'])

        self.motion_mlp_fine.load_state_dict(to_load['motion_mlp'])
        self.traj_basis_fine = (to_load['traj_basis'])


    def load_fine_model(self, filename, load_opt=True, load_scheduler=True):

        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.net_fine_st.load_state_dict(to_load['net_fine_st'])
        self.net_fine_dy.load_state_dict(to_load['net_fine_dy'])

        self.feature_net_fine.load_state_dict(to_load['feature_net_fine'])

        self.motion_mlp_fine.load_state_dict(to_load['motion_mlp_fine'])
        self.traj_basis_fine = (to_load['traj_basis_fine'])

        return to_load['global_step']

    def load_coarse_from_ckpt(self, out_folder,
                              load_opt=True,
                              load_scheduler=True,
                              force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        fpath = ckpts[-1]
        num_steps = self.load_coarse_model(fpath, load_opt, load_scheduler)
        self.load_coarse_model2(fpath, load_opt, load_scheduler)

        print("=========== COARSE num_steps ", num_steps)
        step = num_steps #int(fpath[-10:-4])
        print('Reloading from {}, starting at step={}'.format(fpath, step))

        return step


    def load_fine_from_ckpt(self, out_folder,
                            load_opt=True,
                            load_scheduler=True,
                            force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            num_steps = self.load_fine_model(fpath, load_opt, load_scheduler)
            print("=========== num_steps ", num_steps)
            step = num_steps #int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step

