import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2

from .ray_utils import *

def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return img, depth

class ScannetDataset(Dataset):
    def __init__(self, args, split='train', load_ref=False):
        self.args = args
        self.root_dir = args.datadir
        self.split = split
        downsample = args.imgScale_train if split=='train' else args.imgScale_test
        # assert int(800*downsample)%32 == 0, \
        #     f'image width is {int(800*downsample)}, it should be divisible by 32, you may need to modify the imgScale'
        self.img_wh = (640, 480)
        print(self.img_wh)

        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if not load_ref:
            self.read_meta()

        self.white_back = True

    def read_meta(self):
        splits = ['train', 'test']

        all_imgs = []
        all_poses = []
        all_intrinsics = []
        counts = [0]
        filenames = []
        for s in splits:
            if os.path.exists(os.path.join(self.root_dir, 'transforms_{}.json'.format(s))):

                json_fname =  os.path.join(self.root_dir, 'transforms_{}.json'.format(s))

                with open(json_fname, 'r') as fp:
                    meta = json.load(fp)

                if 'train' in s:
                    near = float(meta['near'])
                    far = float(meta['far'])
               
                imgs = []
                poses = []
                intrinsics = []
                
                for frame in meta['frames']:
                    if len(frame['file_path']) != 0:
                        image_path = os.path.join(self.root_dir, frame['file_path'])
                        img = Image.open(image_path)
                        
                        W, H = img.size

                        x_scale = W/self.img_wh[0]
                        y_scale = H/self.img_wh[1]

                        img = img.resize(self.img_wh, Image.LANCZOS)
                        img = np.array(img)

                        # img, _ = read_files(self.root_dir, frame['file_path'], frame['depth_file_path'])
                        # H, W = img.shape[:2]

                        filenames.append(frame['file_path'])                  
                        imgs.append(img)

                    poses.append(np.array(frame['transform_matrix']))
                    fx, fy, cx, cy = frame['fx']/x_scale, frame['fy']/y_scale, frame['cx']/x_scale, frame['cy']/y_scale
                    intrinsics.append(np.array((fx, fy, cx, cy)))

                counts.append(counts[-1] + len(poses))
                if len(imgs) > 0:
                    all_imgs.append(np.array(imgs))
                all_poses.append(np.array(poses).astype(np.float32))
                all_intrinsics.append(np.array(intrinsics).astype(np.float32))
            else:
                counts.append(counts[-1])

        i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        intrinsics = np.concatenate(all_intrinsics, 0)

        # print(imgs.shape)
        # print(poses.shape)
        # print(intrinsics.shape)

        i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
        i_train, i_test = i_split

        # print(i_train)
        # print(i_test)

        if "train" == self.split:
            idx_to_take = i_train
        else:
            idx_to_take = i_test
        # print(idx_to_take)

        # bounds, common for all scenes
        self.near = near
        self.far = far
        self.bounds = np.array([self.near, self.far])

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []

        for i in idx_to_take:
            pose = poses[i] @ self.blender2opencv
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)            

            image_path = os.path.join(self.root_dir, frame['file_path'])
            self.image_paths += [image_path]

            img = imgs[i]
            H, W = img.shape[:2]
            fx, fy, cx, cy = intrinsics[i]

            img = self.transform(img)  # (4, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 4) RGBA
            self.all_masks += [img[:,-1:]>0]
            self.all_rgbs += [img]


            self.directions = get_ray_directions(H, W, [fx, fy], [cx, cy])  # (h, w, 3)
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near * torch.ones_like(rays_o[:, :1]),
                                         self.far * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)
            self.all_masks += []

        self.poses = np.stack(self.poses)

        if 'train' == self.split:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

        # print(self.poses.shape)
        # print(self.all_rays.shape)
        # print(self.all_rgbs.shape)
        # print(self.all_masks.shape)
        # exit()



    def read_source_views(self, file=f"transforms_train.json", pair_idx=None, device=torch.device("cpu")):

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        near = float(meta['near'])
        far = float(meta['far'])
       
        imgs = []
        poses = []
        intrinsics = []


        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        
        # for i in range(len(meta['frames'])):
        for i in range(3):
            frame = meta['frames'][i]
            if len(frame['file_path']) != 0:
                image_path = os.path.join(self.root_dir, frame['file_path'])
                img = Image.open(image_path)
                
                W, H = img.size

                x_scale = W/self.img_wh[0]
                y_scale = H/self.img_wh[1]

                img = img.resize(self.img_wh, Image.LANCZOS)

                # img, _ = read_files(self.root_dir, frame['file_path'], frame['depth_file_path'])
                img = self.transform(img)            
                imgs.append(src_transform(img))

            c2w = np.array(frame['transform_matrix']) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
            fx, fy, cx, cy = frame['fx']/x_scale, frame['fy']/y_scale, frame['cx']/x_scale, frame['cy']/y_scale

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())
            intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]


        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        near_far_source = [near, far]
        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        # imgs = torch.stack(imgs).float().to(device)
        # proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().to(device)

        # print("Here.")
        # print(imgs.shape)
        # print(proj_mats.shape)
        # print(pose_source['c2ws'].shape)
        # print(pose_source['w2cs'].shape)
        # print(pose_source['intrinsics'].shape)
        # exit()

        return imgs, proj_mats, near_far_source, pose_source

    def load_poses_all(self, file=f"transforms_train.json"):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        c2ws = []
        for i,frame in enumerate(meta['frames']):
            c2ws.append(np.array(frame['transform_matrix']) @ self.blender2opencv)
        return np.stack(c2ws)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            # view, ray_idx = torch.randint(0,len(self.all_rays),(1,)), torch.randperm(self.all_rays.shape[1])[:self.args.batch_size]
            # sample = {'rays': self.all_rays[view,ray_idx],
            #           'rgbs': self.all_rgbs[view,ray_idx]}
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately
            # frame = self.meta['frames'][idx]
            # c2w = torch.FloatTensor(frame['transform_matrix']) @ self.blender2opencv

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample


class BlenderDataset(Dataset):
    def __init__(self, args, split='train', load_ref=False):
        self.args = args
        self.root_dir = args.datadir
        self.split = split
        downsample = args.imgScale_train if split=='train' else args.imgScale_test
        assert int(800*downsample)%32 == 0, \
            f'image width is {int(800*downsample)}, it should be divisible by 32, you may need to modify the imgScale'
        self.img_wh = (int(800*downsample),int(800*downsample))
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        if not load_ref:
            self.read_meta()

        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_train.json"), 'r') as f:
            self.meta = json.load(f)

        # sub select training views from pairing file
        if os.path.exists('configs/pairs.th'):
            name = os.path.basename(self.root_dir)
            self.img_idx = torch.load('configs/pairs.th')[f'{name}_{self.split}']
            self.meta['frames'] = [self.meta['frames'][idx] for idx in self.img_idx]
            print(f'===> {self.split}ing index: {self.img_idx}')

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)


        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        for frame in self.meta['frames']:
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            self.all_masks += [img[:,-1:]>0]
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)

            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near * torch.ones_like(rays_o[:, :1]),
                                         self.far * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)
            self.all_masks += []


        self.poses = np.stack(self.poses)
        if 'train' == self.split:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def read_source_views(self, file=f"transforms_train.json", pair_idx=None, device=torch.device("cpu")):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        w, h = self.img_wh
        focal = 0.5 * 800 / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
        focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # if do not specify source views, load index from pairing file
        if pair_idx is None:
            name = os.path.basename(self.root_dir)
            pair_idx = torch.load('configs/pairs.th')[f'{name}_train'][:3]
            print(f'====> ref idx: {pair_idx}')

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        for i,idx in enumerate(pair_idx):
            frame = meta['frames'][idx]
            c2w = np.array(frame['transform_matrix']) @ self.blender2opencv
            w2c = np.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)

            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            intrinsic = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())
            intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img[:3] * img[-1:] + (1 - img[-1:])  # blend A to RGB
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).float().to(device)

        near_far_source = [2.0,6.0]
        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)

        return imgs, proj_mats, near_far_source, pose_source

    def load_poses_all(self, file=f"transforms_train.json"):
        with open(os.path.join(self.root_dir, file), 'r') as f:
            meta = json.load(f)

        c2ws = []
        for i,frame in enumerate(meta['frames']):
            c2ws.append(np.array(frame['transform_matrix']) @ self.blender2opencv)
        return np.stack(c2ws)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            # view, ray_idx = torch.randint(0,len(self.all_rays),(1,)), torch.randperm(self.all_rays.shape[1])[:self.args.batch_size]
            # sample = {'rays': self.all_rays[view,ray_idx],
            #           'rgbs': self.all_rgbs[view,ray_idx]}
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately
            # frame = self.meta['frames'][idx]
            # c2w = torch.FloatTensor(frame['transform_matrix']) @ self.blender2opencv

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample