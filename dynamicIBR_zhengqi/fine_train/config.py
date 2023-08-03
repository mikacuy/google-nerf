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


import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--rootdir', type=str,
                        default='/home/qw246/S7/code/IBRNet/',
                        help='the path to the project root directory. Replace this path with yours!')
    parser.add_argument('--folder_path', type=str,
                        default='/home/qw246/S7/code/IBRNet/',
                        help='the path to the project root directory. Replace this path with yours!')

    parser.add_argument('--coarse_dir', type=str,
                        default='/home/qw246/S7/code/IBRNet/',
                        help='the path to the project root directory. Replace this path with yours!')

    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument("--local_rank", type=int, default=0, help='rank for distributed training')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument("--decay_base", type=int, default=20, help='decay_base')
    parser.add_argument("--erosion_radius", type=int, default=1, help='erosion_radius')
    parser.add_argument('--mask_static', action='store_true', help='mask_static')

    ########## dataset options ##########
    ## train and eval dataset
    parser.add_argument("--train_dataset", type=str, default='ibrnet_collected',
                        help='the training dataset, should either be a single dataset, '
                             'or multiple datasets connected with "+", for example, ibrnet_collected+llff+spaces')
    parser.add_argument("--dataset_weights", nargs='+', type=float, default=[],
                        help='the weights for training datasets, valid when multiple datasets are used.')
    parser.add_argument("--train_scenes", nargs='+', default=[],
                        help='optional, specify a subset of training scenes from training dataset')
    parser.add_argument('--eval_dataset', type=str, default='llff_test', help='the dataset to evaluate')
    parser.add_argument('--eval_scenes', nargs='+', default=[],
                        help='optional, specify a subset of scenes from eval_dataset to evaluate')
    ## others
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, '
                             'useful for large datasets like deepvoxels or nerf_synthetic')
    parser.add_argument("--render_idx", type=int, default=-1,
                        help='render_idx')
    parser.add_argument("--init_decay_epoch", type=int, default=150,
                        help='init_decay_epoch')


    ########## model options ##########
    ## ray sampling options
    parser.add_argument('--sample_mode', type=str, default='uniform',
                        help='how to sample pixels from images for training:'
                             'uniform|center')
    parser.add_argument('--center_ratio', type=float, default=0.8, help='the ratio of center crop to keep')
    parser.add_argument("--N_rand", type=int, default=32 * 16,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 4,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--thickness", type=int, default=9,
                        help='thickness for ce loss')
    parser.add_argument("--skewness", type=float, default=1,
                        help='skewness for entropy loss')
    parser.add_argument("--lr_multipler", type=float, default=1,
                        help='lr_multipler for static component')
    parser.add_argument("--anneal_cycle", action='store_true',
                        help='anneal_cycle')
    parser.add_argument("--fix_prob", type=float, default=0.5,
                        help='fix_prob')
    parser.add_argument("--sfm_reg", action='store_true',
                        help='sfm_reg')
    
    ## model options
    parser.add_argument('--coarse_feat_dim', type=int, default=32, help="2D feature dimension for coarse level")
    parser.add_argument('--fine_feat_dim', type=int, default=32, help="2D feature dimension for fine level")
    parser.add_argument('--num_source_views', type=int, default=6,
                        help='the number of input source views for each target view')
    parser.add_argument('--num_basis', type=int, default=6,
                        help='the number of basis for motion trajectory')

    parser.add_argument('--rectify_inplane_rotation', action='store_true', help='if rectify inplane rotation')
    parser.add_argument('--coarse_only', action='store_true', help='use coarse network only')
    parser.add_argument("--anti_alias_pooling", type=int, default=1, help='if use anti-alias pooling')
    parser.add_argument("--mask_rgb", type=int, default=0, help='if mask rgb feature')

    parser.add_argument("--decay_rate", type=float, default=10., help='decay_rate')

    ########## checkpoints ##########
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default="",
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_load_opt", action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true',
                        help='do not load scheduler when reloading')

    ########### iterations & learning rate options ##########
    parser.add_argument("--n_iters", type=int, default=250000, help='num of iterations')
    parser.add_argument("--lrate_feature", type=float, default=1e-3, help='learning rate for feature extractor')
    parser.add_argument("--lrate_mlp", type=float, default=5e-4, help='learning rate for mlp')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=50000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument('--w_cycle', type=float, default=0.1, help='the ratio of center crop to keep')
    parser.add_argument('--w_distortion', type=float, default=1e-3, help='the ratio of center crop to keep')
    parser.add_argument('--w_sfm', type=float, default=1e-3, help='the ratio of center crop to keep')

    parser.add_argument('--w_entropy', type=float, default=0.0, help='w_entropy')
    parser.add_argument('--w_constst', type=float, default=0.0, help='w_constst')
    parser.add_argument('--w_disp', type=float, default=5e-2, help='w_disp')
    parser.add_argument('--w_flow', type=float, default=5e-3, help='w_flow')
    parser.add_argument('--w_cross_entropy', type=float, default=1e-3, help='w_flow')
    parser.add_argument('--w_skew_entropy', type=float, default=1e-3, help='w_flow')
    parser.add_argument('--w_sf_sm', type=float, default=0.05, help='w_sf_sm')
    parser.add_argument('--w_sf_zero', type=float, default=0.05, help='w_sf_zero')
    parser.add_argument('--occ_weights_mode', type=int, default=2, help='occ_weights_mode')

    ########## rendering options ##########
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64, help='number of important samples per ray')
    parser.add_argument("--inv_uniform", action='store_true',
                        help='if True, will uniformly sample inverse depths')

    parser.add_argument("--input_dir", action='store_true',
                        help='if True, input global directional with p.e.')
    parser.add_argument("--input_xyz", action='store_true',
                        help='if True, input global xyz with p.e.')

    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='apply the trick to avoid fitting to white background')
    parser.add_argument("--render_stride", type=int, default=1,
                        help='render with large stride for validation to save time')

    ########## logging/saving options ##########
    parser.add_argument("--i_print", type=int, default=100, help='frequency of terminal printout')
    parser.add_argument("--i_img", type=int, default=1000, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    ########## evaluation options ##########
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    return parser
