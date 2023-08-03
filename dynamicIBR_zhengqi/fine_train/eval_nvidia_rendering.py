''''
CUDA_VISIBLE_DEVICES=0,1 python eval_nvidia_rendering.py --config configs_nvidia/eval_jumping_long.txt
CUDA_VISIBLE_DEVICES=2,3 python eval_nvidia_rendering.py --config configs_nvidia/eval_skating_long.txt
python eval_nvidia_rendering.py --config configs_nvidia/eval_playground_long.txt
python eval_nvidia_rendering.py --config configs_nvidia/eval_balloon1_long.txt
python eval_nvidia_rendering.py --config configs_nvidia/eval_balloon2_long.txt

CUDA_VISIBLE_DEVICES=4,5 python eval_nvidia_rendering.py --config configs_nvidia/eval_truck_long.txt
python eval_nvidia_rendering.py --config configs_nvidia/eval_umbrella_long.txt
CUDA_VISIBLE_DEVICES=6,7 python eval_nvidia_rendering.py --config configs_nvidia/eval_playground_long.txt
CUDA_VISIBLE_DEVICES=4,5 python eval_nvidia_rendering.py --config configs_nvidia/eval_balloon1_long.txt
CUDA_VISIBLE_DEVICES=2,3 python eval_nvidia_rendering.py --config configs_nvidia/eval_dynamicFace_long.txt
CUDA_VISIBLE_DEVICES=0,1 python eval_nvidia_rendering.py --config configs_nvidia/eval_balloon2_long.txt

CUDA_VISIBLE_DEVICES=6,7 python eval_nvidia_rendering.py --config configs_nvidia/eval_jumping_long.txt

'''

from torch.utils.data import Dataset
import sys
# sys.path.append('../')
from torch.utils.data import DataLoader
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import get_nearest_pose_ids
from ibrnet.data_loaders.llff_data_utils import load_llff_data, batch_parse_llff_poses
import time
from collections import defaultdict
import skimage.metrics
import math
from tqdm import tqdm

NUM_DYNAMIC_SRC_VIEWS = 7
VIZ_DEBUG = False

class DynamicVideoDataset(Dataset):
    def __init__(self, 
                 render_idx,
                 args,
                 scenes,  
                 **kwargs):

        self.folder_path = args.folder_path #os.path.join('/home/zhengqili/filestore/NSFF/nerf_data')
        self.num_source_views = NUM_DYNAMIC_SRC_VIEWS #args.num_source_views
        self.render_idx = render_idx
        print("num_source_views ", self.num_source_views)
        self.mask_static = args.mask_static

        print("loading {} for rendering".format(scenes))
        assert len(scenes) == 1

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []
        self.h = []
        self.w = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        for i, scene in enumerate(scenes):
            self.scene_path = scene_path = os.path.join(self.folder_path, scene, 'dense')
            _, poses, bds, render_poses, i_test, rgb_files, scale = load_llff_data(scene_path, 
                                                                                   height=288,
                                                                                   num_avg_imgs=12,
                                                                                   render_idx=self.render_idx, 
                                                                                   load_imgs=False)
            near_depth = np.min(bds)
            far_depth = np.max(bds) + 15.
            self.num_frames = len(rgb_files)

            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            h, w = poses[0][:2, -1]
            render_intrinsics, render_c2w_mats = batch_parse_llff_poses(poses)

            i_test = []
            i_val = i_test
            i_train = np.array([i for i in np.arange(len(rgb_files)) if
                                (i not in i_test and i not in i_val)])

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(render_intrinsics)
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in render_intrinsics])

            self.render_poses.extend([c2w_mat for c2w_mat in render_c2w_mats])
            self.render_depth_range.extend([[near_depth, far_depth]]*self.num_frames)
            self.render_train_set_ids.extend([i]*self.num_frames)
            self.h.extend([int(h)]*self.num_frames)
            self.w.extend([int(w)]*self.num_frames)

    def __len__(self):
        return 12

    def __getitem__(self, idx): # idx corresponds to 
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        h, w = self.h[idx], self.w[idx]

        # rescale rendered image
        h *= 0.5
        w *= 0.5
        intrinsics[:2, ...] *= 0.5

        camera = np.concatenate(([h, w], intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        # print("camera ", [h, w], intrinsics)
        # sys.exit()

        gt_img_path = os.path.join(self.scene_path, 
                                   'mv_images', 
                                   '%05d'%self.render_idx, 
                                   'cam%02d.jpg'%(idx + 1))

        nearest_time_ids = [self.render_idx + offset for offset in [1, 2, 3, 0, -1, -2, -3]]
        nearest_pose_ids = nearest_time_ids

        nearest_pose_ids = np.sort(nearest_pose_ids)

        assert len(nearest_pose_ids) == 7

        num_imgs_per_cycle = 12
        static_pose_ids = get_nearest_pose_ids(render_pose,
                                               train_poses,
                                               1000,
                                               tar_id=self.render_idx, # do not include target image itself
                                               angular_dist_method='dist')
        static_id_dict = defaultdict(list)
        for static_pose_id in static_pose_ids:
            # do not include image with the same viewpoint
            if static_pose_id % num_imgs_per_cycle == self.render_idx % num_imgs_per_cycle:
                continue

            static_id_dict[static_pose_id % num_imgs_per_cycle].append(static_pose_id)

        static_pose_ids = []
        for key in static_id_dict:
            min_idx = np.argmin(np.abs(np.array(static_id_dict[key]) - self.render_idx))
            static_pose_ids.append(static_id_dict[key][min_idx])

        static_pose_ids = np.sort(static_pose_ids)

        # assert len(static_pose_ids) == 11

        src_rgbs = []
        src_cameras = []
        for src_idx in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[src_idx]).astype(np.float32) / 255.
            train_pose = train_poses[src_idx]
            train_intrinsics_ = train_intrinsics[src_idx]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)

            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        static_src_rgbs = []
        static_src_cameras = []
        static_src_masks = []

        # load src rgb for static view
        for st_near_id in static_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[st_near_id]).astype(np.float32) / 255.
            train_pose = train_poses[st_near_id]
            train_intrinsics_ = train_intrinsics[st_near_id]

            static_src_rgbs.append(src_rgb)

            # load coarse mask
            if self.mask_static and 3 <= st_near_id < self.num_frames - 3:
                st_mask_path = os.path.join('/'.join(train_rgb_files[st_near_id].split('/')[:-2]), 'coarse_masks', '%05d.png'%st_near_id)
                st_mask = imageio.imread(st_mask_path).astype(np.float32) / 255.
                st_mask = cv2.resize(st_mask, (src_rgb.shape[1], src_rgb.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            else:
                st_mask = np.ones_like(src_rgb[..., 0])

            static_src_masks.append(st_mask)

            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)

            static_src_cameras.append(src_camera)

        static_src_rgbs = np.stack(static_src_rgbs, axis=0)
        static_src_cameras = np.stack(static_src_cameras, axis=0)
        static_src_masks = np.stack(static_src_masks, axis=0)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return {'camera': torch.from_numpy(camera),
                'rgb_path': gt_img_path,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
                'src_cameras': torch.from_numpy(src_cameras).float(),
                'static_src_rgbs': torch.from_numpy(static_src_rgbs[..., :3]).float(),
                'static_src_cameras': torch.from_numpy(static_src_cameras).float(),
                "static_src_masks": torch.from_numpy(static_src_masks).float(),
                'depth_range': depth_range,
                'ref_time': float(self.render_idx / float(self.num_frames)),
                "id": self.render_idx,
                "nearest_pose_ids":nearest_pose_ids
                }


def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2)**2 * mask) / num_valid
    
    if mse == 0:
        return 0 #float('inf')

    return 10 * math.log10(1./mse)


def calculate_ssim(img1, img2, mask):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 1]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    _, ssim_map = skimage.metrics.structural_similarity(img1, img2, multichannel=True, full=True)
    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid

def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

import models

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    test_dataset = DynamicVideoDataset(3, args, scenes=args.eval_scenes)
    args.num_frames = test_dataset.num_frames
    print("args.num_frames ", args.num_frames)
    # Create ibrnet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print('saving results to {}...'.format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, 'renderings')
    print('saving results to {}'.format(out_scene_dir))

    os.makedirs(out_scene_dir, exist_ok=True)
    # os.makedirs(os.path.join(out_scene_dir, 'rgb_debug'), exist_ok=True)
    # os.makedirs(os.path.join(out_scene_dir, 'depth_debug'), exist_ok=True)

    lpips_model = models.PerceptualLoss(model='net-lin',net='alex',
                                        use_gpu=True,version=0.1)

    psnr_list = []
    ssim_list = []
    lpips_list = []

    dy_psnr_list = []
    dy_ssim_list = []
    dy_lpips_list = []

    st_psnr_list = []
    st_ssim_list = []
    st_lpips_list = []

    for img_i in tqdm(range(3, args.num_frames-3)):#args.num_frames-3):
        test_dataset = DynamicVideoDataset(img_i, args, scenes=args.eval_scenes)
        save_prefix = scene_name
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=12, shuffle=False)
        total_num = len(test_loader)
        out_frames = []

        for i, data in enumerate(test_loader):
            gt_img_path = data['rgb_path'][0]
            print("img_i ", img_i, i)

            if img_i % 12 == i:
                continue

            idx = int(data['id'].item())
            start = time.time()

            ref_time_embedding = data['ref_time'].cuda()
            ref_frame_idx = int(data['id'].item())
            ref_time_offset = [int(near_idx - ref_frame_idx) for near_idx in data['nearest_pose_ids'].squeeze().tolist()]

            model.switch_to_eval()
            with torch.no_grad():
                ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
                ray_batch = ray_sampler.get_all()

                cb_featmaps_1, cb_featmaps_2 = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
                ref_featmaps = cb_featmaps_1

                static_src_rgbs = ray_batch['static_src_rgbs'].squeeze(0).permute(0, 3, 1, 2)
                _, static_featmaps = model.feature_net(static_src_rgbs)

                cb_featmaps_1_fine, _ = model.feature_net_fine(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
                ref_featmaps_fine = cb_featmaps_1_fine

                if args.mask_static:
                    static_src_rgbs_ = static_src_rgbs * ray_batch['static_src_masks'].squeeze(0)[:, None, ...]
                else:
                    static_src_rgbs_ = static_src_rgbs

                _, static_featmaps_fine = model.feature_net_fine(static_src_rgbs_)

                ret = render_single_image(frame_idx=(ref_frame_idx, None),
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
                                          is_train=False)

            fine_pred_rgb = ret['outputs_fine_ref']['rgb'].detach().cpu().numpy()
            fine_pred_depth = ret['outputs_fine_ref']['depth'].detach().cpu().numpy()

            valid_mask = np.float32(np.sum(fine_pred_rgb, axis=-1, keepdims=True) > 1e-3)
            valid_mask = np.tile(valid_mask, (1, 1, 3))
            gt_img = cv2.imread(gt_img_path)[:, :, ::-1]
            gt_img = cv2.resize(gt_img, 
                                dsize=(fine_pred_rgb.shape[1], fine_pred_rgb.shape[0]), 
                                interpolation=cv2.INTER_AREA)
            gt_img = np.float32(gt_img) / 255

            fine_pred_rgb_viz = (255 * np.clip(fine_pred_rgb, a_min=0, a_max=1.)).astype(np.uint8)
            gt_img_viz = (255 * np.clip(gt_img, a_min=0, a_max=1.)).astype(np.uint8)
            fine_pred_depth = colorize_np(fine_pred_depth, cmap_name='jet',
                                          range=(np.percentile(fine_pred_depth, 3), np.percentile(fine_pred_depth, 97)))

            # fine_pred_rgb_st = ret['outputs_fine_ref']['rgb_static'].detach().cpu().numpy()
            # fine_pred_rgb_rgb = ret['outputs_fine_ref']['rgb_dy'].detach().cpu().numpy()
            # fine_pred_rgb_st = (255 * np.clip(fine_pred_rgb_st, a_min=0, a_max=1.)).astype(np.uint8)
            # fine_pred_rgb_rgb = (255 * np.clip(fine_pred_rgb_rgb, a_min=0, a_max=1.)).astype(np.uint8)

            imageio.imwrite(os.path.join(out_scene_dir, 'pred_%03d_%03d.png'%(img_i, i)), fine_pred_rgb_viz)
            imageio.imwrite(os.path.join(out_scene_dir, 'gt_%03d_%03d.png'%(img_i, i)), gt_img_viz)
            # sys.exit()


            # gt_img = gt_img * valid_mask
            # fine_pred_rgb = fine_pred_rgb * valid_mask

            # dynamic_mask = valid_mask
            # ssim = calculate_ssim(gt_img, 
            #                       fine_pred_rgb, 
            #                       dynamic_mask)
            # psnr = calculate_psnr(gt_img, 
            #                       fine_pred_rgb, 
            #                       dynamic_mask)

            # gt_img_0 = im2tensor(gt_img).cuda()
            # fine_pred_rgb_0 = im2tensor(fine_pred_rgb).cuda()
            # dynamic_mask_0 = torch.Tensor(dynamic_mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

            # lpips = lpips_model.forward(gt_img_0, 
            #                             fine_pred_rgb_0, 
            #                             dynamic_mask_0).item()
            # print(psnr, ssim, lpips)
            # psnr_list.append(psnr)
            # ssim_list.append(ssim)
            # lpips_list.append(lpips)

            # dynamic_mask_path = os.path.join(test_dataset.scene_path, 
            #                                 'mv_masks', 
            #                                 '%05d'%img_i, 
            #                                 'cam%02d.png'%(i + 1))     

            # dynamic_mask = np.float32(cv2.imread(dynamic_mask_path) > 1e-3)#/255.
            # dynamic_mask = cv2.resize(dynamic_mask, 
            #                           dsize=(gt_img.shape[1], gt_img.shape[0]), 
            #                           interpolation=cv2.INTER_NEAREST)
            # # dynamic_mask = dynamic_mask #np.ones_like(gt_img[..., 0:3])
            # dynamic_mask_0 = torch.Tensor(dynamic_mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
            # dynamic_ssim = calculate_ssim(gt_img, 
            #                               fine_pred_rgb, 
            #                               dynamic_mask)
            # dynamic_psnr = calculate_psnr(gt_img, 
            #                               fine_pred_rgb, 
            #                               dynamic_mask)
            # dynamic_lpips = lpips_model.forward(gt_img_0, 
            #                                     fine_pred_rgb_0, 
            #                                     dynamic_mask_0).item()
            # print(dynamic_psnr, dynamic_ssim, dynamic_lpips)

            # dy_psnr_list.append(dynamic_psnr)
            # dy_ssim_list.append(dynamic_ssim)
            # dy_lpips_list.append(dynamic_lpips)

            # static_mask = 1 - dynamic_mask
            # static_mask_0 = torch.Tensor(static_mask[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
            # static_ssim = calculate_ssim(gt_img, 
            #                              fine_pred_rgb, 
            #                              static_mask)
            # static_psnr = calculate_psnr(gt_img, 
            #                              fine_pred_rgb, 
            #                              static_mask)
            # static_lpips = lpips_model.forward(gt_img_0, 
            #                                    fine_pred_rgb_0, 
            #                                    static_mask_0).item()
            # print(static_psnr, static_ssim, static_lpips)

            # st_psnr_list.append(static_psnr)
            # st_ssim_list.append(static_ssim)
            # st_lpips_list.append(static_lpips)

