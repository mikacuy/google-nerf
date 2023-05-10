import torch
import os, sys
from os import path
from config import get_config
from model import MipNeRF
import imageio
from datasets import get_dataloader, cycle
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals

from skimage.metrics import structural_similarity
from lpips import LPIPS
import cv2
import numpy as np


img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.full((1,), 10., device=x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to16b = lambda x : ((2**16 - 1) * np.clip(x,0,1)).astype(np.uint16)


### Seed ###
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
############

######
class MeanTracker(object):
    def __init__(self):
        self.reset()

    def add(self, input, weight=1.):
        for key, l in input.items():
            if not key in self.mean_dict:
                self.mean_dict[key] = 0
            self.mean_dict[key] = (self.mean_dict[key] * self.total_weight + l) / (self.total_weight + weight)
        self.total_weight += weight

    def has(self, key):
        return (key in self.mean_dict)

    def get(self, key):
        return self.mean_dict[key]
    
    def as_dict(self):
        return self.mean_dict
        
    def reset(self):
        self.mean_dict = dict()
        self.total_weight = 0
    
    def print(self, f=None):
        for key, l in self.mean_dict.items():
            if f is not None:
                print("{}: {}".format(key, l), file=f)
            else:
                print("{}: {}".format(key, l))
######

def visualize(config):
    data = get_dataloader(config.dataset_name, config.base_dir, split="test", factor=config.factor, shuffle=False, mode="test_time")
    # data = get_dataloader(config.dataset_name, config.base_dir, split="render", factor=config.factor, shuffle=False)

    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
        mode=config.mode,
        correct_hier=config.correct_hier
    )

    print("Using correct hierarchical sampling: "+str(config.correct_hier))

    model.load_state_dict(torch.load(os.path.join(config.log_dir, config.model_weight_path)))
    model.eval()

    count = len(data)
    H = data.h
    W= data.w

    rgbs_res = torch.empty(count, 3, H, W)
    # rgbs0_res = torch.empty(count, 3, H, W)
    target_rgbs_res = torch.empty(count, 3, H, W)
    # depths_res = torch.empty(count, 1, H, W)
    # depths0_res = torch.empty(count, 1, H, W)
    # target_depths_res = torch.empty(count, 1, H, W)
    # target_valid_depths_res = torch.empty(count, 1, H, W, dtype=bool)
    all_depth_maps = []

    mean_metrics = MeanTracker()
    # mean_depth_metrics = MeanTracker() # track separately since they are not always available
    lpips_alex = LPIPS()

    img_idx = 1
    for rays, image_gt in tqdm(data):
        print("Render image {}/{}".format(img_idx, count))
        target = image_gt


        rgb, dist, acc = model.render_image_testset(rays, data.h, data.w, chunks=2048)
        depth_map = to8b(visualize_depth(dist.cpu().numpy(), acc.cpu().numpy(), data.near, data.far))
        
        rgb = torch.clamp(rgb, 0, 1)

        ## There are some nan values in the output for some reason (for model with imporatance sampling).... --> making this equivalent to the background
        # rgb = torch.where(torch.isnan(rgb), torch.ones_like(rgb), rgb)

        # print(rgb[rgb!=1.])
        # print(torch.isnan(rgb).any())
        # print(torch.isnan(target).any())
        # exit()

        img_loss = img2mse(rgb, target)

        psnr = mse2psnr(img_loss)

        ssim = structural_similarity(rgb.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)
        lpips = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]        
        
        print(img_idx)
        print("PSNR: {}".format(psnr))
        print("LPIPS: {}".format(lpips[0, 0, 0]))
        print("SSIM: {}".format(ssim))
        print()

        rgbs_res[img_idx-1] = rgb.clamp(0., 1.).permute(2, 0, 1).cpu()
        target_rgbs_res[img_idx-1] = target.permute(2, 0, 1).cpu()
        all_depth_maps.append(depth_map)

        metrics = {"img_loss" : img_loss.item(), "psnr" : psnr.item(), "ssim" : ssim, "lpips" : lpips[0, 0, 0],}
        mean_metrics.add(metrics)

        img_idx +=1

    result_dir = os.path.join(config.log_dir, "test_images_" + config.scene)
    os.makedirs(result_dir, exist_ok=True)

    for n, (rgb, depth, gt_rgb) in enumerate(zip(rgbs_res.permute(0, 2, 3, 1).cpu().numpy(), \
            all_depth_maps, target_rgbs_res.permute(0, 2, 3, 1).cpu().numpy())):

        # write rgb
        # cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".jpg"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".png"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))

        cv2.imwrite(os.path.join(result_dir, str(n) + "_gt" + ".png"), cv2.cvtColor(to8b(gt_rgb), cv2.COLOR_RGB2BGR))

        # write depth
        cv2.imwrite(os.path.join(result_dir, str(n) + "_d" + ".png"), depth)

    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        mean_metrics.print(f)
    mean_metrics.print()



if __name__ == "__main__":
    config = get_config()
    visualize(config)
