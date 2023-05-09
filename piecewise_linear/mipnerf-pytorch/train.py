import os.path
import shutil
from config import get_config
from scheduler import MipLRDecay
from loss import NeRFLoss, mse_to_psnr
from model import MipNeRF
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from os import path
from datasets import get_dataloader, cycle
import numpy as np
from tqdm import tqdm

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


def train_model(config):
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")

    data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="train", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device, near=config.set_near_plane)))
    eval_data = None
    if config.do_eval:
        eval_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device, near=config.set_near_plane)))

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
        color_mode=config.color_mode,
        correct_hier=config.correct_hier
    )

    print("Using correct hierarchical sampling: "+str(config.correct_hier))

    optimizer = optim.AdamW(model.parameters(), lr=config.lr_init, weight_decay=config.weight_decay)
    if config.continue_training:
        model.load_state_dict(torch.load(model_save_path))
        optimizer.load_state_dict(torch.load(optimizer_save_path))

    scheduler = MipLRDecay(optimizer, lr_init=config.lr_init, lr_final=config.lr_final, max_steps=config.max_steps, lr_delay_steps=config.lr_delay_steps, lr_delay_mult=config.lr_delay_mult)
    loss_func = NeRFLoss(config.coarse_weight_decay)
    model.train()

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(path.join(config.log_dir, config.expname), exist_ok=True)
    shutil.rmtree(path.join(config.log_dir, config.expname, 'train'), ignore_errors=True)
    logger = tb.SummaryWriter(path.join(config.log_dir, config.expname, 'train'), flush_secs=1)

    for step in tqdm(range(0, config.max_steps)):
    # for step in range(0, config.max_steps):
        rays, pixels = next(data)
        comp_rgb, _, _ = model(rays)
        pixels = pixels.to(config.device)

        # Compute loss and update model weights.
        loss_val, psnr = loss_func(comp_rgb, pixels, rays.lossmult.to(config.device))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        psnr = psnr.detach().cpu().numpy()
        logger.add_scalar('train/loss', float(loss_val.detach().cpu().numpy()), global_step=step)
        logger.add_scalar('train/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
        logger.add_scalar('train/fine_psnr', float(psnr[-1]), global_step=step)
        logger.add_scalar('train/avg_psnr', float(np.mean(psnr)), global_step=step)
        logger.add_scalar('train/lr', float(scheduler.get_last_lr()[-1]), global_step=step)

        if step % config.save_every == 0:
            if eval_data:
                del rays
                del pixels
                psnr = eval_model(config, model, eval_data)
                psnr = psnr.detach().cpu().numpy()
                logger.add_scalar('eval/coarse_psnr', float(np.mean(psnr[:-1])), global_step=step)
                logger.add_scalar('eval/fine_psnr', float(psnr[-1]), global_step=step)
                logger.add_scalar('eval/avg_psnr', float(np.mean(psnr)), global_step=step)

            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), optimizer_save_path)

        if step % config.print_every == 0:
            print(f"[TRAIN] Iter: {step} Loss: {float(loss_val.detach().cpu().numpy())}  COARSE PSNR: {float(np.mean(psnr[:-1]))}  FINE PSNR: {float(psnr[-1])}")

    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)


    ### Eval after training ###
    data = get_dataloader(config.dataset_name, config.base_dir, split="test", factor=config.factor, shuffle=False, mode="test_time")
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
    # lpips_alex = None

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

        # ssim = structural_similarity(rgb.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)
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
        # metrics = {"img_loss" : img_loss.item(), "psnr" : psnr.item(), "ssim" : ssim}
        mean_metrics.add(metrics)

        img_idx +=1

    result_dir = os.path.join(config.log_dir, config.expname, "test_images_" + config.scene)
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

    ###########################


def eval_model(config, model, data):
    model.eval()
    rays, pixels = next(data)
    with torch.no_grad():
        comp_rgb, _, _ = model(rays)
    pixels = pixels.to(config.device)
    model.train()
    return torch.tensor([mse_to_psnr(torch.mean((rgb - pixels[..., :3])**2)) for rgb in comp_rgb])


if __name__ == "__main__":
    config = get_config()
    train_model(config)
