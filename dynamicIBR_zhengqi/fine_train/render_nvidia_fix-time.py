''''
python render_nvidia_test.py --config configs/eval_kid-running.txt
python render_nvidia_test.py --config configs/eval_playground.txt
python render_nvidia_test.py --config configs/eval_ballon1.txt
python render_nvidia_test.py --config configs/eval_ballon2.txt
python render_nvidia_test.py --config configs/eval_umbrella.txt

'''

from torch.utils.data import Dataset
import sys
# sys.path.append('../')
from torch.utils.data import DataLoader
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image, render_single_image_test
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import get_nearest_pose_ids
from ibrnet.data_loaders.llff_data_utils import load_llff_data, batch_parse_llff_poses
import time
from collections import defaultdict

class DynamicVideoDataset(Dataset):
    def __init__(self, args,
                 scenes,  # 'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'
                 **kwargs):

        self.folder_path = args.folder_path #os.path.join('/home/zhengqili/filestore/NSFF/nerf_data')
        self.num_source_views = args.num_source_views
        self.render_idx = args.render_idx
        print("num_source_views ", self.num_source_views)

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
            scene_path = os.path.join(self.folder_path, scene, 'dense')
            _, poses, bds, render_poses, i_test, rgb_files, scale = load_llff_data(scene_path, 
                                                                                   render_idx=self.render_idx, 
                                                                                   load_imgs=False)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            self.num_frames = len(rgb_files)

            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            h, w = poses[0][:2, -1]
            render_intrinsics, render_c2w_mats = batch_parse_llff_poses(render_poses)

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
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)
            self.h.extend([int(h)]*num_render)
            self.w.extend([int(w)]*num_render)

    def __len__(self):
        return len(self.render_poses)

    def __getitem__(self, idx):
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        h, w = self.h[idx], self.w[idx]
        camera = np.concatenate(([h, w], intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        nearest_time_ids = [self.render_idx + offset for offset in [1, 2, 3, -1, -2, -3]]

        # id_render = -1
        nearest_dist_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                self.num_source_views,
                                                tar_id=self.render_idx,
                                                angular_dist_method='dist')

        nearest_dist_ids = nearest_dist_ids[:self.num_source_views]

        nearest_pose_ids = [self.render_idx]
        for nearest_dist_id in nearest_dist_ids:
            if nearest_dist_id in nearest_time_ids:
                nearest_pose_ids.append(nearest_dist_id)

        if len(nearest_pose_ids) < 6:
            for time_id in nearest_time_ids:
                if time_id not in nearest_pose_ids:
                    nearest_pose_ids.append(time_id)

        nearest_pose_ids = nearest_pose_ids[:6]
        nearest_pose_ids = np.sort(nearest_pose_ids)

        assert len(nearest_pose_ids) == 6

        num_imgs_per_cycle = 12
        static_pose_ids = get_nearest_pose_ids(render_pose,
                                               train_poses,
                                               10,
                                               tar_id=self.render_idx,
                                               angular_dist_method='dist')
        static_id_dict = defaultdict(list)
        # nearest_pose_ids_mod = [id_ % num_imgs_per_cycle for id_ in nearest_pose_ids]
        for static_pose_id in static_pose_ids:
            if static_pose_id % num_imgs_per_cycle == idx % num_imgs_per_cycle:
                continue

            static_id_dict[static_pose_id % num_imgs_per_cycle].append(static_pose_id)

        static_pose_ids = []
        for key in static_id_dict:
            min_idx = np.argmin(np.abs(np.array(static_id_dict[key]) - idx))
            static_pose_ids.append(static_id_dict[key][min_idx])

        static_pose_ids = np.sort(static_pose_ids)

        nearest_pose_ids = np.concatenate([nearest_pose_ids, static_pose_ids])

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
        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return {'camera': torch.from_numpy(camera),
                'rgb_path': '',
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                'ref_time': float(self.render_idx / float(self.num_frames)),
                "id": self.render_idx,
                "nearest_pose_ids":nearest_pose_ids
                }

NUM_DYNAMIC_SRC_VIEWS = 6


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    test_dataset = DynamicVideoDataset(args, scenes=args.eval_scenes)
    args.num_frames = test_dataset.num_frames

    # Create ibrnet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print('saving results to {}...'.format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}_{:03d}'.format(scene_name, model.start_step, args.render_idx), 'videos')
    print('saving results to {}'.format(out_scene_dir))

    os.makedirs(out_scene_dir, exist_ok=True)
    os.makedirs(os.path.join(out_scene_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(out_scene_dir, 'depth'), exist_ok=True)

    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    out_frames = []
    crop_ratio = 0.1


    num_dy_views = args.dy_time_window * 2
    assert num_dy_views == 6
    
    model.switch_to_eval()

    for i, data in enumerate(test_loader):
        start = time.time()
        # src_rgbs = data['src_rgbs'][0].cpu().numpy()
        # averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        # imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(i)), averaged_img)
        ref_time_embedding = data['ref_time'].cuda()
        ref_frame_idx = int(data['id'].item())
        ref_time_offset = [int(near_idx - ref_frame_idx) for near_idx in data['nearest_pose_ids'].squeeze().tolist()[:num_dy_views]]

        # print("ref_time_offset ", ref_time_offset)
        # sys.exit()

        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()

            ref_featmaps, _ = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)[:num_dy_views])
            _, static_featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2)[num_dy_views:])

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
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      featmaps=(ref_featmaps, None, static_featmaps),
                                      is_train=False)
            # torch.cuda.empty_cache()

        coarse_pred_rgb = ret['outputs_coarse_ref']['rgb'].detach().cpu()
        coarse_pred_rgb_st = ret['outputs_coarse_ref']['rgb_static'].detach().cpu()
        coarse_pred_rgb_rgb = ret['outputs_coarse_ref']['rgb_dy'].detach().cpu()

        coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        coarse_pred_rgb_st = (255 * np.clip(coarse_pred_rgb_st.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        coarse_pred_rgb_dy = (255 * np.clip(coarse_pred_rgb_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)

        comb_coarse_rgb = np.concatenate([coarse_pred_rgb, coarse_pred_rgb_st, coarse_pred_rgb_dy], axis=1)

        h, w = coarse_pred_rgb.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        # coarse_pred_rgb = coarse_pred_rgb[crop_h:h -crop_h, crop_w:w - crop_w, :]
        imageio.imwrite(os.path.join(out_scene_dir, 'rgb', '{}.png'.format(i)), comb_coarse_rgb)

        coarse_pred_depth = ret['outputs_coarse_ref']['depth'].detach().cpu()
        # coarse_pred_depth = coarse_pred_depth[crop_h:h -crop_h, crop_w:w - crop_w]

        # imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(i)),
                        # (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
        coarse_pred_depth_colored = colorize_np(coarse_pred_depth, cmap_name='jet',
                                                range=tuple(data['depth_range'].squeeze().numpy()))


        imageio.imwrite(os.path.join(out_scene_dir, 'depth', '{}.png'.format(i)),
                        (255 * coarse_pred_depth_colored).astype(np.uint8))

        # coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'].detach().cpu(), dim=-1)
        # coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
        # imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_coarse.png'.format(i)),
        #                 coarse_acc_map_colored)

        # if ret['outputs_fine'] is not None:
        #     fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
        #     fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        #     imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_fine.png'.format(i)), fine_pred_rgb)
        #     fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
        #     imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_fine.png'.format(i)),
        #                     (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
        #     fine_pred_depth_colored = colorize_np(fine_pred_depth,
        #                                           range=tuple(data['depth_range'].squeeze().cpu().numpy()))
        #     imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_fine.png'.format(i)),
        #                     (255 * fine_pred_depth_colored).astype(np.uint8))
        #     fine_acc_map = torch.sum(ret['outputs_fine']['weights'].detach().cpu(), dim=-1)
        #     fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
        #     imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_fine.png'.format(i)),
        #                     fine_acc_map_colored)
        # else:
        #     fine_pred_rgb = None

        # out_frame = fine_pred_rgb if fine_pred_rgb is not None else coarse_pred_rgb
        # h, w = averaged_img.shape[:2]
        # crop_h = int(h * crop_ratio)
        # crop_w = int(w * crop_ratio)
        # # crop out image boundaries
        # out_frame = out_frame[crop_h:h - crop_h, crop_w:w - crop_w, :]
        # out_frames.append(out_frame)

        print('frame {} completed, {}'.format(i, time.time() - start))

    # imageio.mimwrite(os.path.join(extra_out_dir, '{}.mp4'.format(scene_name)), out_frames, fps=20, quality=8)
