''''
python render_nvidia_space-time.py --config configs_nvidia/eval_jumping_long.txt
python render_nvidia_space-time.py --config configs_nvidia/eval_skating_long.txt
python render_nvidia_space-time.py --config configs_nvidia/eval_playground_long.txt
python render_nvidia_space-time.py --config configs_nvidia/eval_balloon1_long.txt
python render_nvidia_space-time.py --config configs_nvidia/eval_balloon2_long.txt

python render_nvidia_space-time.py --config configs_nvidia/eval_truck_long.txt
python render_nvidia_space-time.py --config configs_nvidia/eval_umbrella_long.txt
python render_nvidia_space-time.py --config configs_nvidia/eval_dynamicFace_long.txt

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

NUM_DYNAMIC_SRC_VIEWS = 7

class DynamicVideoDataset(Dataset):
    def __init__(self, args,
                 scenes,  # 'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'
                 **kwargs):

        self.folder_path = args.folder_path #os.path.join('/home/zhengqili/filestore/NSFF/nerf_data')
        self.num_source_views = NUM_DYNAMIC_SRC_VIEWS #args.num_source_views
        self.render_idx = 8
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
                                                                                   height=288,
                                                                                   num_avg_imgs=12,
                                                                                   render_idx=self.render_idx, 
                                                                                   load_imgs=False)
            near_depth = np.min(bds)
            far_depth = np.max(bds) + 15.
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
            self.render_depth_range.extend([[near_depth, far_depth]]*self.num_frames)
            self.render_train_set_ids.extend([i]*self.num_frames)
            self.h.extend([int(h)]*self.num_frames)
            self.w.extend([int(w)]*self.num_frames)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # idx = (idx % (self.num_frames - 6) + 3)
        # idx = 10
        # idx = 31
        render_pose = self.render_poses[idx % len(self.render_poses)]
        intrinsics = self.render_intrinsics[idx % len(self.render_poses)]
        depth_range = self.render_depth_range[idx % len(self.render_poses)]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        h, w = self.h[idx], self.w[idx]
        camera = np.concatenate(([h, w], intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)


        nearest_time_ids = [idx + offset for offset in [1, 2, 3, 0, -1, -2, -3]]
        nearest_pose_ids = nearest_time_ids

        nearest_pose_ids = np.sort(nearest_pose_ids)
        nearest_pose_ids = nearest_pose_ids[:NUM_DYNAMIC_SRC_VIEWS]

        assert len(nearest_pose_ids) == 7

        num_imgs_per_cycle = 12
        static_pose_ids = get_nearest_pose_ids(render_pose,
                                               train_poses,
                                               1000,
                                               tar_id=-1,
                                               angular_dist_method='dist')
        static_id_dict = defaultdict(list)
        # nearest_pose_ids_mod = [id_ % num_imgs_per_cycle for id_ in nearest_pose_ids]
        for static_pose_id in static_pose_ids:
            # if static_pose_id % num_imgs_per_cycle == idx % num_imgs_per_cycle:
                # continue

            static_id_dict[static_pose_id % num_imgs_per_cycle].append(static_pose_id)

        static_pose_ids = []
        for key in static_id_dict:
            min_idx = np.argmin(np.abs(np.array(static_id_dict[key]) - idx))
            static_pose_ids.append(static_id_dict[key][min_idx])

        static_pose_ids = np.sort(static_pose_ids)

        assert len(static_pose_ids) == 12

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
        # load src rgb for static view
        for st_near_id in static_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[st_near_id]).astype(np.float32) / 255.
            train_pose = train_poses[st_near_id]
            train_intrinsics_ = train_intrinsics[st_near_id]

            static_src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)

            static_src_cameras.append(src_camera)

        static_src_rgbs = np.stack(static_src_rgbs, axis=0)
        static_src_cameras = np.stack(static_src_cameras, axis=0)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return {'camera': torch.from_numpy(camera),
                'rgb_path': '',
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
                'src_cameras': torch.from_numpy(src_cameras).float(),
                'static_src_rgbs': torch.from_numpy(static_src_rgbs[..., :3]).float(),
                'static_src_cameras': torch.from_numpy(static_src_cameras).float(),
                'depth_range': depth_range,
                'ref_time': float(idx / float(self.num_frames)),
                "id": idx,
                "nearest_pose_ids":nearest_pose_ids
                }



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
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step), 'videos')
    print('saving results to {}'.format(out_scene_dir))

    os.makedirs(out_scene_dir, exist_ok=True)
    os.makedirs(os.path.join(out_scene_dir, 'rgb'), exist_ok=True)
    os.makedirs(os.path.join(out_scene_dir, 'depth'), exist_ok=True)

    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    total_num = len(test_loader)
    out_frames = []
    crop_ratio = 0.02

    # try:
    for i, data in enumerate(test_loader):
        idx = int(data['id'].item())

        if idx < 3 or idx >= total_num - 3:
            continue

        start = time.time()
        # src_rgbs = data['src_rgbs'][0].cpu().numpy()
        # averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        # imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(i)), averaged_img)
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
            _, static_featmaps_fine = model.feature_net_fine(static_src_rgbs)

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

        fine_pred_rgb = ret['outputs_fine_ref']['rgb'].detach().cpu()
        fine_pred_rgb_st = ret['outputs_fine_ref']['rgb_static'].detach().cpu()
        fine_pred_rgb_rgb = ret['outputs_fine_ref']['rgb_dy'].detach().cpu()

        fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        fine_pred_rgb_st = (255 * np.clip(fine_pred_rgb_st.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        fine_pred_rgb_dy = (255 * np.clip(fine_pred_rgb_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)

        h, w = fine_pred_rgb.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)

        fine_pred_rgb = fine_pred_rgb[crop_h:h-crop_h, crop_w:w-crop_w, ...]
        fine_pred_rgb_st = fine_pred_rgb_st[crop_h:h-crop_h, crop_w:w-crop_w, ...]
        fine_pred_rgb_dy = fine_pred_rgb_dy[crop_h:h-crop_h, crop_w:w-crop_w, ...]

        comb_fine_rgb = np.concatenate([fine_pred_rgb, fine_pred_rgb_st, fine_pred_rgb_dy], axis=1)

        out_frames.append(comb_fine_rgb)

        # fine_pred_rgb = fine_pred_rgb[crop_h:h -crop_h, crop_w:w - crop_w, :]
        imageio.imwrite(os.path.join(out_scene_dir, 'rgb', '{}.png'.format(i)), comb_fine_rgb)

        fine_pred_depth = ret['outputs_fine_ref']['depth'].detach().cpu()

        fine_pred_depth = fine_pred_depth[crop_h:h-crop_h, crop_w:w-crop_w, ...]

        fine_pred_depth_colored = colorize_np(fine_pred_depth, cmap_name='jet',
                                                range=tuple(data['depth_range'].squeeze().numpy()))

        imageio.imwrite(os.path.join(out_scene_dir, 'depth', '{}.png'.format(i)),
                        (255 * fine_pred_depth_colored).astype(np.uint8))

        print('frame {} completed, {}'.format(i, time.time() - start))
    # except:
    #     imageio.mimwrite(os.path.join(extra_out_dir, '{}.mp4'.format(args.expname)), out_frames, fps=25)
    #     imageio.mimwrite(os.path.join(extra_out_dir, '{}.gif'.format(args.expname)), out_frames, fps=25)
