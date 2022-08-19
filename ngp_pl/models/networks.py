import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import vren
from .custom_functions import TruncExp
import numpy as np

from .rendering import NEAR_DISTANCE


class NGP(nn.Module):
    def __init__(self, scale, num_levels=16):
        super().__init__()

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = num_levels; F_ = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F_} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3,
                n_output_dims=16,
                encoding_config=
                # ####InstantNGP
                # {
                #     "otype": "HashGrid",
                #     "n_levels": L,
                #     "n_features_per_level": F_,
                #     "log2_hashmap_size": log2_T,
                #     "base_resolution": N_min,
                #     "per_level_scale": b,
                # },
                #### Normal NeRF
                {
                    "otype": "Frequency", # Component type.
                    "n_frequencies": 12   # Number of frequencies (sin & cos)
                                          # per encoded dimension.
                },               
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

        self.sigma_act = TruncExp.apply

    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        h = self.xyz_encoder(x)
        sigmas = self.sigma_act(h[:, 0]).float()
        if return_feat: return sigmas, h
        return sigmas

    def forward(self, x, d):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        d /= torch.norm(d, dim=-1, keepdim=True)
        d = self.dir_encoder((d+1)/2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        return sigmas, rgbs

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M):
        """
        Sample both M uniform and occupied cells (per cascade)
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>0)[:, 0]
            rand_idx = torch.randint(len(indices2), (M,),
                                     device=self.density_grid.device)
            indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        # print(poses.shape)
        
        # x.permute(*torch.arange(x.ndim -1, -1, -1)) <<--- .mT
        # w2c_R = poses[:, :3, :3].permute(*torch.arange(poses[:, :3, :3].ndim -1, -1, -1)) # (N, 3, 3) batch transpose
        # w2c_R = poses[:, :3, :3].mT # (N, 3, 3) batch transpose

        w2c_R = poses[:, :3, :3].permute(0,2,1) # (N, 3, 3) batch transpose
        
        # print(w2c_R.shape)
        # print(poses[:, :3, 3:].shape)

        w2c_T = torch.bmm(-w2c_R, poses[:, :3, 3:]) # (N, 3, 1)

        # w2c_T = -w2c_R@poses[:, :3, 3:] # (N, 3, 1)
        
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                
                xyzs_w = xyzs_w.unsqueeze(0).repeat(w2c_R.shape[0], 1, 1)
                xyzs_c = torch.bmm(w2c_R, xyzs_w) + w2c_T # (N, 3, chunk)

                # xyzs_c = w2c_R @ xyzs_w + w2c_T # (N, 3, chunk)

                uvd = K @ xyzs_c # (N, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is visible by at least one camera
                covered_by_any_cam = covered_by_cam.any(0)
                too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = covered_by_any_cam&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)

        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        # My own implementation. To avoid floaters, I compare the density
        # with its neighbors (3x3x3). If it is the max among them, it is possible
        # (not always) that it is a floater in empty space, so decay the density more.
        if erode:
            grid = self.density_grid.view(
                self.cascades, self.grid_size, self.grid_size, self.grid_size)
            maxpool = F.max_pool3d(grid, kernel_size=3, stride=1, padding=1)
            local_max = (grid==maxpool)&(maxpool>0)
            self.density_grid[local_max.view(self.cascades, -1)] *= decay

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)
