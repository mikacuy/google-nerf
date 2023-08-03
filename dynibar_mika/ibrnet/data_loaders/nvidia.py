import os
import numpy as np
import imageio
import torch
import sys
sys.path.append('../')
from torch.utils.data import Dataset
from .data_utils import get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses, load_llff_data_multicam
from .flow_utils import flow_to_image, warp_flow
from collections import defaultdict
import random
import skimage.morphology
import cv2

ucsd_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:3, 11:4, 12:5, 13:6}

class NvidiaDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.folder_path = args.folder_path #os.path.join('/home/zhengqili/filestore/NSFF/nerf_data')
        self.fix_prob = args.fix_prob
        self.erosion_radius = args.erosion_radius
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        for i, scene in enumerate(scenes):
            self.scene_path = os.path.join(self.folder_path, scene, 'dense')      
            # scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, i_test, rgb_files, scale = load_llff_data(self.scene_path, 
                num_avg_imgs=12, height=288, load_imgs=False)
            near_depth = np.min(bds)
            far_depth = np.max(bds) + 15.

            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            i_test = [] #np.arange(poses.shape[0])[::self.args.llffhold]
            i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                (j not in i_test and j not in i_test)])

            if mode == 'train':
                i_render = i_train
            else:
                i_render = i_test

            self.num_frames = len(rgb_files)
            self.scale = scale
            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def read_optical_flow(self, basedir, img_i, start_frame, fwd, interval):
        import os
        flow_dir = os.path.join(basedir, 'flow_i%d'%interval)

        if fwd:
          fwd_flow_path = os.path.join(flow_dir, 
                                      '%05d_fwd.npz'%(start_frame + img_i))
          fwd_data = np.load(fwd_flow_path)#, (w, h))
          fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
          fwd_mask = np.float32(fwd_mask)  
          
          return fwd_flow, fwd_mask
        else:
          bwd_flow_path = os.path.join(flow_dir, 
                                      '%05d_bwd.npz'%(start_frame + img_i))

          bwd_data = np.load(bwd_flow_path)#, (w, h))
          bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
          bwd_mask = np.float32(bwd_mask)

          return bwd_flow, bwd_mask

    def compute_flow_from_depth(self, p_2d_grid_1, mvs_depth, K, T_1_G, T_2_G):

        T_2_1 = np.dot(T_2_G, np.linalg.inv(T_1_G))
        R_2_1 = T_2_1[:3, :3]
        t_2_1 = T_2_1[:3, 3:]

        p_2d_vec_1 = np.reshape(p_2d_grid_1, (-1,2)).T
        p_2d_vec_1 = np.concatenate((p_2d_vec_1, np.ones_like(p_2d_vec_1[:1,:])), axis=0)

        bearing_1 = np.dot(np.linalg.inv(K), p_2d_vec_1)

        p_3d_mvs_1 = bearing_1 * np.tile( np.reshape(mvs_depth, (1, -1)), (3,1))
        p_2d_mvs_2 = np.dot(K, np.dot(R_2_1, p_3d_mvs_1) + t_2_1)
        p_2d_vec_2 = p_2d_mvs_2 / np.tile(p_2d_mvs_2[2:, :], (3, 1))

        mvs_fwd_flow = p_2d_vec_2 - p_2d_vec_1
        mvs_fwd_flow = mvs_fwd_flow[:2,:].T
        mvs_fwd_flow = np.reshape(mvs_fwd_flow, (mvs_depth.shape[0], mvs_depth.shape[1], 2))
        # mvs_fwd_flow = mvs_fwd_flow * np.tile(np.expand_dims(np.float32(mvs_depth > 0), axis=-1), (1,1,2))

        return mvs_fwd_flow


    def __len__(self):
        return self.num_frames #* 100000 if self.mode == 'train' else len(self.render_rgb_files)

    def set_global_step(self, globa_step):
        self.globa_step = globa_step

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        idx = np.random.randint(3, self.num_frames - 3)

        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]


        # load motion mask
        # mask_path = rgb_file.replace('images_', 'masks_')
        mask_path = rgb_file.replace(rgb_file.split("/")[-2], 'coarse_masks')
        motion_mask = imageio.imread(mask_path).astype(np.float32) / 255.

        if self.erosion_radius > 0:
            motion_mask = skimage.morphology.erosion(motion_mask > 1e-3, 
                                                     skimage.morphology.disk(self.erosion_radius))

        static_mask_path = os.path.join('/'.join(rgb_file.split('/')[:-2]), 'static_masks', '%d.png'%idx)
        sfm_mask = 1. - imageio.imread(static_mask_path).astype(np.float32) / 255.

        # load mono-depth 
        disp_path = os.path.join('/'.join(rgb_file.split("/")[:-4]), 'Depths', rgb_file.split("/")[-4]+"_disp",'disp', rgb_file.split('/')[-1][:-4] + '.npy')  
        disp = np.load(disp_path) / self.scale

        assert disp.shape[0:2] == rgb.shape[0:2] 
        assert motion_mask.shape[0:2] == rgb.shape[0:2] 

        # print("motion_mask ", motion_mask.shape)
        # sys.exit()

        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(19, 6))
        # plt.subplot(1, 3, 1)
        # plt.imshow(rgb)
        # plt.subplot(1, 3, 2)
        # plt.imshow(disp, cmap='gray') 
        # plt.subplot(1, 3, 3)
        # plt.imshow(motion_mask, cmap='gray')
        # plt.tight_layout()
        # plt.savefig('depth_flow_mask_%d.png'%idx)
        # sys.exit()

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        # Navie view selection based on time
        # nearest_pose_ids = [idx + offset for offset in [1, 2, 3, -1, -2, -3]]
        nearest_pose_ids = [idx + offset for offset in [-3, -2, -1, 1, 2, 3]]

        max_step = min(3, self.current_epoch // self.args.init_decay_epoch + 1)
        bootstrap = (self.current_epoch // self.args.init_decay_epoch == 0)

        anchor_pool = [i for i in range(1, max_step+1)] + [-i for i in range(1, max_step+1)]
        anchor_idx = idx + anchor_pool[np.random.choice(len(anchor_pool))]
        anchor_nearest_pose_ids = []

        anchor_camera = np.concatenate((list(img_size), self.render_intrinsics[anchor_idx].flatten(),
                                       self.render_poses[anchor_idx].flatten())).astype(np.float32)

        for offset in [3, 2, 1, 0, -1, -2, -3]:
            if (anchor_idx + offset) < 0 or (anchor_idx + offset) >= len(train_rgb_files) or (anchor_idx + offset) == idx :
                continue
            anchor_nearest_pose_ids.append((anchor_idx + offset))

        # assert id_render not in nearest_pose_ids
        # occasionally include input image
        # if np.random.choice([0, 1], p=[0.995, 0.005]):
        #     nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = idx

        # # # occasinally include render image for anchor time index
        if np.random.choice([0, 1], p=[0.99, 0.01]):
            anchor_nearest_pose_ids[np.random.choice(len(anchor_nearest_pose_ids))] = idx

        nearest_pose_ids = np.sort(nearest_pose_ids)
        anchor_nearest_pose_ids = np.sort(anchor_nearest_pose_ids)

        flows, masks = [], []

        # load optical flow
        curr_offsets = [-1, 1]
        for offset in curr_offsets: ### Only have 1 flow neighbor for now
        # for ii in range(len(nearest_pose_ids)):
            # offset = nearest_pose_ids[ii] - idx
            try:
                flow, mask = self.read_optical_flow(self.scene_path, 
                                                    idx, 
                                                    start_frame=0, 
                                                    fwd=True if offset > 0 else False, 
                                                    interval=np.abs(offset))
                # print("nearest_pose_ids[ii] ", nearest_pose_ids[ii], idx, offset)
                # print(np.max(flow), np.min(flow))
            except:
                raise ValueError
                flow = np.zeros((rgb.shape[0], rgb.shape[1], 2))
                mask = np.zeros((rgb.shape[0], rgb.shape[1]))
                                                               
            flows.append(flow)
            masks.append(mask)

        flows = np.stack(flows)
        masks = np.stack(masks)

        if False:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            imageio.imwrite('im_%d.png'%(idx), rgb)

            for ii in range(len(nearest_pose_ids)):
                offset = nearest_pose_ids[ii] - idx
                print(idx, offset)
                rgb_source = imageio.imread(self.render_rgb_files[idx + offset]).astype(np.float32) / 255.
                warped_im0_from_p1 = warp_flow(rgb_source, flows[ii])
                # warped_im0_from_p1_mask = warped_im0_from_p1 * masks[ii, ..., None]
                imageio.imwrite('im_%d_offset_%d.png'%(idx, offset), np.clip(warped_im0_from_p1, 0., 1.) )
                # imageio.imwrite('im_%d_offset_%d-mask.png'%(idx, offset), np.clip(warped_im0_from_p1_mask, 0., 1.) )
                # imageio.imwrite('im_%d_offset_%d-flow.png'%(idx, offset), np.clip(flow_to_image(flows[ii])/255., 0., 1.) )

            plt.figure(figsize=(19, 6))
            plt.subplot(2, 4, 1)
            plt.imshow(rgb)
            plt.subplot(2, 4, 5)
            plt.imshow(disp, cmap='gray') 
            plt.subplot(2, 4, 2)
            plt.imshow(flow_to_image(flows[0])/255. * masks[0, ..., None])
            plt.subplot(2, 4, 3)
            plt.imshow(flow_to_image(flows[1])/255. * masks[1, ..., None])
            plt.subplot(2, 4, 4)
            plt.imshow(flow_to_image(flows[2])/255. * masks[2, ..., None])
            plt.subplot(2, 4, 6)
            plt.imshow(flow_to_image(flows[3])/255. * masks[3, ..., None])
            plt.subplot(2, 4, 7)
            plt.imshow(flow_to_image(flows[4])/255. * masks[4, ..., None])
            plt.subplot(2, 4, 8)
            plt.imshow(flow_to_image(flows[5])/255. * masks[5, ..., None])

            plt.tight_layout()
            plt.savefig('depth_flow_%d.png'%idx)
            sys.exit()

        # load src rgb for ref view
        # LOAD SRC RGB FOR REF VIEW, STATIC COMPONENT, CURRENTLY THIS IS FOR NVIDIA DATASET SETUP
        num_imgs_per_cycle = 12
        # static_pose_ids = get_nearest_pose_ids(render_pose,
        #                                        train_poses,
        #                                        num_select=1000,
        #                                        tar_id=idx,
        #                                        angular_dist_method='dist')
        static_pose_ids = get_nearest_pose_ids(render_pose,
                                               train_poses,
                                               tar_id=idx,
                                               angular_dist_method='dist')
        
        static_id_dict = defaultdict(list)

        if bootstrap or random.random() > self.fix_prob: # bootstrap stage 
            # 1
            MAX_RANGE = 12
            for static_pose_id in static_pose_ids:
                # pass images with the same viewpoint as ref images
                if np.abs(static_pose_id - idx) >= MAX_RANGE \
                or (static_pose_id % num_imgs_per_cycle == idx % num_imgs_per_cycle):
                    continue

                static_id_dict[static_pose_id % num_imgs_per_cycle].append(static_pose_id)

            static_pose_ids = []
            for key in static_id_dict:
                static_pose_ids.append(static_id_dict[key][np.random.randint(0, len(static_id_dict[key]))])

        else: # fixed source view now
            for static_pose_id in static_pose_ids:
                if static_pose_id % num_imgs_per_cycle == idx % num_imgs_per_cycle:
                    continue

                static_id_dict[static_pose_id % num_imgs_per_cycle].append(static_pose_id)

            static_pose_ids = []
            for key in static_id_dict:
                min_idx = np.argmin(np.abs(np.array(static_id_dict[key]) - idx))
                static_pose_ids.append(static_id_dict[key][min_idx])

        static_pose_ids = np.sort(static_pose_ids)

        src_rgbs = []
        src_cameras = []

        for near_id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[near_id]).astype(np.float32) / 255.
            train_pose = train_poses[near_id]
            train_intrinsics_ = train_intrinsics[near_id]

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
        
        # # ==================== DEBUG ======================
        if False:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            print("camera ", camera)
            print("disp ", np.max(disp), np.min(disp))

            debug_idx = 1
            [xx, yy] = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            pixel_2d_grid = np.stack( (xx.astype(np.float), yy.astype(np.float)), axis=-1)
            K = src_cameras[debug_idx, 2:2+16].reshape(4, 4)
            T_2_G = np.linalg.inv(src_cameras[debug_idx, 18:18+16].reshape(4, 4))
            T_1_G = np.linalg.inv(camera[18:18+16].reshape(4, 4))

            flow_from_depth = self.compute_flow_from_depth(pixel_2d_grid, 1./disp, K[:3, :3], T_1_G, T_2_G)
            print("flow_from_depth ", flow_from_depth.shape)

            plt.figure(figsize=(12, 6))
            plt.subplot(2, 2, 1)
            plt.imshow(rgb)
            plt.subplot(2, 2, 3)
            plt.imshow(flow_to_image(flows[debug_idx])/255.) 
            plt.subplot(2, 2, 4)
            plt.imshow(flow_to_image(flow_from_depth)/255.)
            plt.subplot(2, 2, 2)
            plt.imshow(np.linalg.norm(flow_from_depth - flows[debug_idx], axis=-1) < 1.5, cmap='gray')
            plt.tight_layout()
            plt.savefig('depth-flow-diff_%d.png'%idx)
            plt.close()
            sys.exit()  

        # load src rgb for anchor view
        anchor_src_rgbs = []
        anchor_src_cameras = []

        for near_id in anchor_nearest_pose_ids:
            # print("near_id ", near_id, len(train_rgb_files))

            src_rgb = imageio.imread(train_rgb_files[near_id]).astype(np.float32) / 255.
            train_pose = train_poses[near_id]
            train_intrinsics_ = train_intrinsics[near_id]

            anchor_src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            anchor_src_cameras.append(src_camera)

        anchor_src_rgbs = np.stack(anchor_src_rgbs, axis=0)
        anchor_src_cameras = np.stack(anchor_src_cameras, axis=0)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5]).float()

        return {'id': idx,
                'anchor_id':anchor_idx,
                "num_frames":self.num_frames,
                'ref_time': float(idx / float(self.num_frames)),
                'anchor_time':float(anchor_idx / float(self.num_frames)),
                "nearest_pose_ids":torch.from_numpy(np.array(nearest_pose_ids)),
                "anchor_nearest_pose_ids":torch.from_numpy(np.array(anchor_nearest_pose_ids)),
                'rgb': torch.from_numpy(rgb[..., :3]).float(),
                'disp': torch.from_numpy(disp).float(),
                'motion_mask': torch.from_numpy(motion_mask).float(),
                'static_mask': torch.from_numpy(sfm_mask).float(),
                'flows': torch.from_numpy(flows).float(),
                'masks': torch.from_numpy(masks).float(),
                'camera': torch.from_numpy(camera).float(),
                "anchor_camera":torch.from_numpy(anchor_camera).float(),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
                'src_cameras': torch.from_numpy(src_cameras).float(),
                'static_src_rgbs': torch.from_numpy(static_src_rgbs[..., :3]).float(),
                'static_src_cameras': torch.from_numpy(static_src_cameras).float(),
                'anchor_src_rgbs': torch.from_numpy(anchor_src_rgbs[..., :3]).float(),
                'anchor_src_cameras': torch.from_numpy(anchor_src_cameras).float(),
                'depth_range': depth_range
                }


class NvidiaDataset_MultiCam(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        '''
        camera_indices : select which cameras to use from the nvidia data
        '''
        self.folder_path = args.folder_path #os.path.join('/home/zhengqili/filestore/NSFF/nerf_data')
        self.fix_prob = args.fix_prob
        self.erosion_radius = args.erosion_radius
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        for i, scene in enumerate(scenes):
            self.scene_path = os.path.join(self.folder_path, scene, 'dense')      
            # scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, i_test, rgb_files, scale = load_llff_data_multicam(self.scene_path, args.camera_indices, 
                num_avg_imgs=12, load_imgs=False)
            near_depth = np.min(bds)
            far_depth = np.max(bds) + 15.

            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            i_test = [] #np.arange(poses.shape[0])[::self.args.llffhold]
            i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                (j not in i_test and j not in i_test)])

            if mode == 'train':
                i_render = i_train
            else:
                i_render = i_test

            self.num_frames = len(rgb_files)
            self.scale = scale
            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def read_optical_flow(self, basedir, img_i, start_frame, fwd, interval):
        import os
        flow_dir = os.path.join(basedir, 'flow_i%d'%interval)

        if fwd:
          fwd_flow_path = os.path.join(flow_dir, 
                                      '%05d_fwd.npz'%(start_frame + img_i))
          fwd_data = np.load(fwd_flow_path)#, (w, h))
          fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
          fwd_mask = np.float32(fwd_mask)  
          
          return fwd_flow, fwd_mask
        else:
          bwd_flow_path = os.path.join(flow_dir, 
                                      '%05d_bwd.npz'%(start_frame + img_i))

          bwd_data = np.load(bwd_flow_path)#, (w, h))
          bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
          bwd_mask = np.float32(bwd_mask)

          return bwd_flow, bwd_mask

    def compute_flow_from_depth(self, p_2d_grid_1, mvs_depth, K, T_1_G, T_2_G):

        T_2_1 = np.dot(T_2_G, np.linalg.inv(T_1_G))
        R_2_1 = T_2_1[:3, :3]
        t_2_1 = T_2_1[:3, 3:]

        p_2d_vec_1 = np.reshape(p_2d_grid_1, (-1,2)).T
        p_2d_vec_1 = np.concatenate((p_2d_vec_1, np.ones_like(p_2d_vec_1[:1,:])), axis=0)

        bearing_1 = np.dot(np.linalg.inv(K), p_2d_vec_1)

        p_3d_mvs_1 = bearing_1 * np.tile( np.reshape(mvs_depth, (1, -1)), (3,1))
        p_2d_mvs_2 = np.dot(K, np.dot(R_2_1, p_3d_mvs_1) + t_2_1)
        p_2d_vec_2 = p_2d_mvs_2 / np.tile(p_2d_mvs_2[2:, :], (3, 1))

        mvs_fwd_flow = p_2d_vec_2 - p_2d_vec_1
        mvs_fwd_flow = mvs_fwd_flow[:2,:].T
        mvs_fwd_flow = np.reshape(mvs_fwd_flow, (mvs_depth.shape[0], mvs_depth.shape[1], 2))
        # mvs_fwd_flow = mvs_fwd_flow * np.tile(np.expand_dims(np.float32(mvs_depth > 0), axis=-1), (1,1,2))

        return mvs_fwd_flow


    def __len__(self):
        return self.num_frames #* 100000 if self.mode == 'train' else len(self.render_rgb_files)

    def set_global_step(self, globa_step):
        self.globa_step = globa_step

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        idx = np.random.randint(3, self.num_frames - 3)

        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]


        # load motion mask
        # mask_path = rgb_file.replace('images_', 'masks_')
        mask_path = rgb_file.replace(rgb_file.split("/")[-2], 'coarse_masks')
        motion_mask = imageio.imread(mask_path).astype(np.float32) / 255.

        if self.erosion_radius > 0:
            motion_mask = skimage.morphology.erosion(motion_mask > 1e-3, 
                                                     skimage.morphology.disk(self.erosion_radius))

        static_mask_path = os.path.join('/'.join(rgb_file.split('/')[:-2]), 'static_masks', '%d.png'%idx)
        sfm_mask = 1. - imageio.imread(static_mask_path).astype(np.float32) / 255.

        # load mono-depth 
        disp_path = os.path.join('/'.join(rgb_file.split("/")[:-4]), 'Depths', rgb_file.split("/")[-4]+"_disp",'disp', rgb_file.split('/')[-1][:-4] + '.npy')  
        disp = np.load(disp_path) / self.scale

        assert disp.shape[0:2] == rgb.shape[0:2] 
        assert motion_mask.shape[0:2] == rgb.shape[0:2] 

        # print("motion_mask ", motion_mask.shape)
        # sys.exit()

        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(19, 6))
        # plt.subplot(1, 3, 1)
        # plt.imshow(rgb)
        # plt.subplot(1, 3, 2)
        # plt.imshow(disp, cmap='gray') 
        # plt.subplot(1, 3, 3)
        # plt.imshow(motion_mask, cmap='gray')
        # plt.tight_layout()
        # plt.savefig('depth_flow_mask_%d.png'%idx)
        # sys.exit()

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        # Navie view selection based on time
        # nearest_pose_ids = [idx + offset for offset in [1, 2, 3, -1, -2, -3]]
        nearest_pose_ids = [idx + offset for offset in [-3, -2, -1, 1, 2, 3]]

        max_step = min(3, self.current_epoch // self.args.init_decay_epoch + 1)
        bootstrap = (self.current_epoch // self.args.init_decay_epoch == 0)

        anchor_pool = [i for i in range(1, max_step+1)] + [-i for i in range(1, max_step+1)]
        anchor_idx = idx + anchor_pool[np.random.choice(len(anchor_pool))]
        anchor_nearest_pose_ids = []

        anchor_camera = np.concatenate((list(img_size), self.render_intrinsics[anchor_idx].flatten(),
                                       self.render_poses[anchor_idx].flatten())).astype(np.float32)

        for offset in [3, 2, 1, 0, -1, -2, -3]:
            if (anchor_idx + offset) < 0 or (anchor_idx + offset) >= len(train_rgb_files) or (anchor_idx + offset) == idx :
                continue
            anchor_nearest_pose_ids.append((anchor_idx + offset))

        # assert id_render not in nearest_pose_ids
        # occasionally include input image
        # if np.random.choice([0, 1], p=[0.995, 0.005]):
        #     nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = idx

        # # # occasinally include render image for anchor time index
        if np.random.choice([0, 1], p=[0.99, 0.01]):
            anchor_nearest_pose_ids[np.random.choice(len(anchor_nearest_pose_ids))] = idx

        nearest_pose_ids = np.sort(nearest_pose_ids)
        anchor_nearest_pose_ids = np.sort(anchor_nearest_pose_ids)

        flows, masks = [], []

        # load optical flow
        curr_offsets = [-1, 1]
        for offset in curr_offsets: ### Only have 1 flow neighbor for now
        # for ii in range(len(nearest_pose_ids)):
            # offset = nearest_pose_ids[ii] - idx
            try:
                flow, mask = self.read_optical_flow(self.scene_path, 
                                                    idx, 
                                                    start_frame=0, 
                                                    fwd=True if offset > 0 else False, 
                                                    interval=np.abs(offset))
                # print("nearest_pose_ids[ii] ", nearest_pose_ids[ii], idx, offset)
                # print(np.max(flow), np.min(flow))
            except:
                raise ValueError
                flow = np.zeros((rgb.shape[0], rgb.shape[1], 2))
                mask = np.zeros((rgb.shape[0], rgb.shape[1]))
                                                               
            flows.append(flow)
            masks.append(mask)

        flows = np.stack(flows)
        masks = np.stack(masks)

        if False:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            imageio.imwrite('im_%d.png'%(idx), rgb)

            for ii in range(len(nearest_pose_ids)):
                offset = nearest_pose_ids[ii] - idx
                print(idx, offset)
                rgb_source = imageio.imread(self.render_rgb_files[idx + offset]).astype(np.float32) / 255.
                warped_im0_from_p1 = warp_flow(rgb_source, flows[ii])
                # warped_im0_from_p1_mask = warped_im0_from_p1 * masks[ii, ..., None]
                imageio.imwrite('im_%d_offset_%d.png'%(idx, offset), np.clip(warped_im0_from_p1, 0., 1.) )
                # imageio.imwrite('im_%d_offset_%d-mask.png'%(idx, offset), np.clip(warped_im0_from_p1_mask, 0., 1.) )
                # imageio.imwrite('im_%d_offset_%d-flow.png'%(idx, offset), np.clip(flow_to_image(flows[ii])/255., 0., 1.) )

            plt.figure(figsize=(19, 6))
            plt.subplot(2, 4, 1)
            plt.imshow(rgb)
            plt.subplot(2, 4, 5)
            plt.imshow(disp, cmap='gray') 
            plt.subplot(2, 4, 2)
            plt.imshow(flow_to_image(flows[0])/255. * masks[0, ..., None])
            plt.subplot(2, 4, 3)
            plt.imshow(flow_to_image(flows[1])/255. * masks[1, ..., None])
            plt.subplot(2, 4, 4)
            plt.imshow(flow_to_image(flows[2])/255. * masks[2, ..., None])
            plt.subplot(2, 4, 6)
            plt.imshow(flow_to_image(flows[3])/255. * masks[3, ..., None])
            plt.subplot(2, 4, 7)
            plt.imshow(flow_to_image(flows[4])/255. * masks[4, ..., None])
            plt.subplot(2, 4, 8)
            plt.imshow(flow_to_image(flows[5])/255. * masks[5, ..., None])

            plt.tight_layout()
            plt.savefig('depth_flow_%d.png'%idx)
            sys.exit()

        # load src rgb for ref view
        # LOAD SRC RGB FOR REF VIEW, STATIC COMPONENT, CURRENTLY THIS IS FOR NVIDIA DATASET SETUP
        num_imgs_per_cycle = 12
        # static_pose_ids = get_nearest_pose_ids(render_pose,
        #                                        train_poses,
        #                                        num_select=1000,
        #                                        tar_id=idx,
        #                                        angular_dist_method='dist')
        static_pose_ids = get_nearest_pose_ids(render_pose,
                                               train_poses,
                                               tar_id=idx,
                                               angular_dist_method='dist')
        
        static_id_dict = defaultdict(list)

        if bootstrap or random.random() > self.fix_prob: # bootstrap stage 
            # 1
            MAX_RANGE = 12
            for static_pose_id in static_pose_ids:
                # pass images with the same viewpoint as ref images
                if np.abs(static_pose_id - idx) >= MAX_RANGE \
                or (static_pose_id % num_imgs_per_cycle == idx % num_imgs_per_cycle):
                    continue

                static_id_dict[static_pose_id % num_imgs_per_cycle].append(static_pose_id)

            static_pose_ids = []
            for key in static_id_dict:
                static_pose_ids.append(static_id_dict[key][np.random.randint(0, len(static_id_dict[key]))])

        else: # fixed source view now
            for static_pose_id in static_pose_ids:
                if static_pose_id % num_imgs_per_cycle == idx % num_imgs_per_cycle:
                    continue

                static_id_dict[static_pose_id % num_imgs_per_cycle].append(static_pose_id)

            static_pose_ids = []
            for key in static_id_dict:
                min_idx = np.argmin(np.abs(np.array(static_id_dict[key]) - idx))
                static_pose_ids.append(static_id_dict[key][min_idx])

        static_pose_ids = np.sort(static_pose_ids)

        src_rgbs = []
        src_cameras = []

        for near_id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[near_id]).astype(np.float32) / 255.
            train_pose = train_poses[near_id]
            train_intrinsics_ = train_intrinsics[near_id]

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
        
        # # ==================== DEBUG ======================
        if False:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            print("camera ", camera)
            print("disp ", np.max(disp), np.min(disp))

            debug_idx = 1
            [xx, yy] = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            pixel_2d_grid = np.stack( (xx.astype(np.float), yy.astype(np.float)), axis=-1)
            K = src_cameras[debug_idx, 2:2+16].reshape(4, 4)
            T_2_G = np.linalg.inv(src_cameras[debug_idx, 18:18+16].reshape(4, 4))
            T_1_G = np.linalg.inv(camera[18:18+16].reshape(4, 4))

            flow_from_depth = self.compute_flow_from_depth(pixel_2d_grid, 1./disp, K[:3, :3], T_1_G, T_2_G)
            print("flow_from_depth ", flow_from_depth.shape)

            plt.figure(figsize=(12, 6))
            plt.subplot(2, 2, 1)
            plt.imshow(rgb)
            plt.subplot(2, 2, 3)
            plt.imshow(flow_to_image(flows[debug_idx])/255.) 
            plt.subplot(2, 2, 4)
            plt.imshow(flow_to_image(flow_from_depth)/255.)
            plt.subplot(2, 2, 2)
            plt.imshow(np.linalg.norm(flow_from_depth - flows[debug_idx], axis=-1) < 1.5, cmap='gray')
            plt.tight_layout()
            plt.savefig('depth-flow-diff_%d.png'%idx)
            plt.close()
            sys.exit()  

        # load src rgb for anchor view
        anchor_src_rgbs = []
        anchor_src_cameras = []

        for near_id in anchor_nearest_pose_ids:
            # print("near_id ", near_id, len(train_rgb_files))

            src_rgb = imageio.imread(train_rgb_files[near_id]).astype(np.float32) / 255.
            train_pose = train_poses[near_id]
            train_intrinsics_ = train_intrinsics[near_id]

            anchor_src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            anchor_src_cameras.append(src_camera)

        anchor_src_rgbs = np.stack(anchor_src_rgbs, axis=0)
        anchor_src_cameras = np.stack(anchor_src_cameras, axis=0)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5]).float()

        return {'id': idx,
                'anchor_id':anchor_idx,
                "num_frames":self.num_frames,
                'ref_time': float(idx / float(self.num_frames)),
                'anchor_time':float(anchor_idx / float(self.num_frames)),
                "nearest_pose_ids":torch.from_numpy(np.array(nearest_pose_ids)),
                "anchor_nearest_pose_ids":torch.from_numpy(np.array(anchor_nearest_pose_ids)),
                'rgb': torch.from_numpy(rgb[..., :3]).float(),
                'disp': torch.from_numpy(disp).float(),
                'motion_mask': torch.from_numpy(motion_mask).float(),
                'static_mask': torch.from_numpy(sfm_mask).float(),
                'flows': torch.from_numpy(flows).float(),
                'masks': torch.from_numpy(masks).float(),
                'camera': torch.from_numpy(camera).float(),
                "anchor_camera":torch.from_numpy(anchor_camera).float(),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]).float(),
                'src_cameras': torch.from_numpy(src_cameras).float(),
                'static_src_rgbs': torch.from_numpy(static_src_rgbs[..., :3]).float(),
                'static_src_cameras': torch.from_numpy(static_src_cameras).float(),
                'anchor_src_rgbs': torch.from_numpy(anchor_src_rgbs[..., :3]).float(),
                'anchor_src_cameras': torch.from_numpy(anchor_src_cameras).float(),
                'depth_range': depth_range
                }