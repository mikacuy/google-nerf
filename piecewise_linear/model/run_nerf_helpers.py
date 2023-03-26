import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.full((1,), 10., device=x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
to16b = lambda x : ((2**16 - 1) * np.clip(x,0,1)).astype(np.uint16)

def precompute_quadratic_samples(near, far, num_samples):
    # normal parabola between 0.1 and 1, shifted and scaled to have y range between near and far
    start = 0.1
    x = torch.linspace(0, 1, num_samples)
    c = near
    a = (far - near)/(1. + 2. * start)
    b = 2. * start * a
    return a * x.pow(2) + b * x + c

def is_not_in_expected_distribution(depth_mean, depth_var, depth_measurement_mean, depth_measurement_std):
    delta_greater_than_expected = ((depth_mean - depth_measurement_mean).abs() - depth_measurement_std) > 0.
    var_greater_than_expected = depth_measurement_std.pow(2) < depth_var
    return torch.logical_or(delta_greater_than_expected, var_greater_than_expected)

def compute_depth_loss(depth_map, z_vals, weights, target_depth, target_valid_depth):
    pred_mean = depth_map[target_valid_depth]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = ((z_vals[target_valid_depth] - pred_mean.unsqueeze(-1)).pow(2) * weights[target_valid_depth]).sum(-1) + 1e-5
    target_mean = target_depth[..., 0][target_valid_depth]
    target_std = target_depth[..., 1][target_valid_depth]
    apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)
    pred_mean = pred_mean[apply_depth_loss]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = pred_var[apply_depth_loss]
    target_mean = target_mean[apply_depth_loss]
    target_std = target_std[apply_depth_loss]
    f = nn.GaussianNLLLoss(eps=0.001)
    return float(pred_mean.shape[0]) / float(target_valid_depth.shape[0]) * f(pred_mean, target_mean, pred_var)


################################
##### For MiDaS-based loss #####
################################
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def compute_monosdf_styleloss(pred_depth, target_depth, mask=None):
    if mask is None:
        mask = torch.ones_like(pred_depth)

    # target_depth = target_depth.squeeze(0)
    pred_depth = pred_depth.unsqueeze(0).unsqueeze(-1)
    mask = mask.unsqueeze(0).unsqueeze(-1)

    # print(pred_depth.shape)
    # print(target_depth.shape)

    scale, shift = compute_scale_and_shift(pred_depth, target_depth, mask)

    # print(scale.shape)
    # print(shift.shape)

    prediction_ssi = scale.view(-1, 1, 1) * pred_depth + shift.view(-1, 1, 1)
    # print(prediction_ssi.shape)

    loss = torch.norm(prediction_ssi - target_depth, p=2, dim=-1)
    loss = torch.mean(loss)

    # print(loss)
    # print(loss.shape)

    return loss

################################

def get_space_carving_idx(pred_depth, target_hypothesis, is_joint=False, mask=None, norm_p=2, threshold=0.0):
    H, W, n_points = pred_depth.shape
    num_hypothesis = target_hypothesis.shape[0]

    # print(target_hypothesis.squeeze())

    target_hypothesis_repeated = target_hypothesis.repeat(1, 1, 1, n_points)

    ## L2 distance
    # distances = torch.sqrt((pred_depth - target_hypothesis_repeated)**2)
    distances = torch.norm(pred_depth.unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=norm_p, dim=-1)

    if mask is not None:
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)

    if is_joint:
        ### Take the mean for all points on all rays, hypothesis is chosen per image
        total_loss = torch.mean(torch.mean(torch.mean(distances, axis=-1), axis=-1), axis=-1)
        best_idx = torch.argmin(total_loss, dim=0)
        best_idx = best_idx.unsqueeze(-1).unsqueeze(-1).repeat(H, W)

    else:
        ### Each ray selects a hypothesis
        total_loss = torch.mean(distances, axis=-1) ## Loss per ray
        best_idx = torch.argmin(total_loss, dim=0) 

    # print(best_idx)
    # print(best_idx.shape)
    # exit()

    return best_idx

def get_space_carving_idx_corrected(pred_depth, target_hypothesis, is_joint=False, mask=None, norm_p=2, threshold=0.0):
    H, W, n_points = pred_depth.shape
    num_hypothesis = target_hypothesis.shape[0]

    # print(target_hypothesis.squeeze())

    target_hypothesis_repeated = target_hypothesis.repeat(1, 1, 1, n_points)

    ## L2 distance
    # distances = torch.sqrt((pred_depth - target_hypothesis_repeated)**2)
    distances = torch.norm(pred_depth.unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=norm_p, dim=-1)

    if mask is not None:
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)


    if is_joint:
        ### Take the mean for all points on all rays, hypothesis is chosen per image
        total_loss = torch.mean(torch.mean(distances, axis=1), axis=1)
        best_idx = torch.argmin(total_loss, dim=0)
        best_idx = best_idx.unsqueeze(0).unsqueeze(0).repeat(H, W, 1)

    else:
        ### Each ray selects a hypothesis
        best_idx = torch.argmin(distances, dim=0) ## Loss per ray

    return best_idx


def compute_space_carving_loss(pred_depth, target_hypothesis, is_joint=False, mask=None, norm_p=2, threshold=0.0):
    n_rays, n_points = pred_depth.shape
    num_hypothesis = target_hypothesis.shape[0]

    # print(target_hypothesis.squeeze())

    target_hypothesis_repeated = target_hypothesis.repeat(1, 1, n_points)

    ## L2 distance
    # distances = torch.sqrt((pred_depth - target_hypothesis_repeated)**2)
    distances = torch.norm(pred_depth.unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=norm_p, dim=-1)

    if mask is not None:
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)


    if is_joint:
        ### Take the mean for all points on all rays, hypothesis is chosen per image
        total_loss = torch.mean(torch.mean(distances, axis=-1), axis=-1)
        best_loss = torch.min(total_loss, dim=0)[0]

    else:
        ### Each ray selects a hypothesis
        total_loss = torch.mean(distances, axis=-1) ## Loss per ray

        best_hyp = torch.min(total_loss, dim=0)[0]
        best_loss = torch.mean(best_hyp)  

    return best_loss

def compute_space_carving_loss_corrected(pred_depth, target_hypothesis, is_joint=False, mask=None, norm_p=2, threshold=0.0):
    n_rays, n_points = pred_depth.shape
    num_hypothesis = target_hypothesis.shape[0]

    if target_hypothesis.shape[-1] == 1:
        ### In the case where there is no caching of quantiles
        target_hypothesis_repeated = target_hypothesis.repeat(1, 1, n_points)
    else:
        ### Each quantile here already picked a hypothesis
        target_hypothesis_repeated = target_hypothesis

    ## L2 distance
    # distances = torch.sqrt((pred_depth - target_hypothesis_repeated)**2)
    distances = torch.norm(pred_depth.unsqueeze(-1) - target_hypothesis_repeated.unsqueeze(-1), p=norm_p, dim=-1)

    if mask is not None:
        mask = mask.unsqueeze(0).repeat(distances.shape[0],1).unsqueeze(-1)
        distances = distances * mask

    if threshold > 0:
        distances = torch.where(distances < threshold, torch.tensor([0.0]).to(distances.device), distances)

    if is_joint:
        ### Take the mean for all points on all rays, hypothesis is chosen per image
        quantile_mean = torch.mean(distances, axis=1) ## mean for each quantile, averaged across all rays
        samples_min = torch.min(quantile_mean, axis=0)[0]
        loss =  torch.mean(samples_min, axis=-1)


    else:
        ### Each ray selects a hypothesis
        best_hyp = torch.min(distances, dim=0)[0]   ## for each sample pick a hypothesis
        ray_mean = torch.mean(best_hyp, dim=-1) ## average across samples
        loss = torch.mean(ray_mean)  

    return loss


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * np.pi * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs).to(inputs.device) for fn in self.embed_fns], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_cam=0, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_cam = input_ch_cam
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + input_ch_cam + W, W//2, activation="relu")])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views + self.input_ch_cam], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, F.softplus(alpha, beta=10)], -1)
        else:
            outputs = self.output_linear(h)
            outputs = torch.cat([outputs[..., :3], F.softplus(outputs[..., 3:], beta=10)], -1)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


class NeRF_deeper_viewlinear(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_cam=0, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF_deeper_viewlinear, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_cam = input_ch_cam
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + input_ch_cam + W, W//2, activation="relu")])
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + input_ch_cam + W, W//2)] + [nn.Linear(W//2, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views + self.input_ch_cam], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, F.softplus(alpha, beta=10)], -1)
        else:
            outputs = self.output_linear(h)
            outputs = torch.cat([outputs[..., :3], F.softplus(outputs[..., 3:], beta=10)], -1)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


class NeRF_deeper_viewlinear2(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_cam=0, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF_deeper_viewlinear2, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_cam = input_ch_cam
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + input_ch_cam + W, W//2, activation="relu")])
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2 + input_ch_cam, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    def forward(self, x):
        input_pts, input_views, input_cam = torch.split(x, [self.input_ch, self.input_ch_views, self.input_ch_cam], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            
            ### linear layers for viewing direction and camera latent
            h = self.views_linears[0](h)
            h = F.relu(h)

            ## Concat input camera
            h = torch.cat([h, input_cam], -1)
            h = self.views_linears[1](h)
            h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, F.softplus(alpha, beta=10)], -1)
        else:
            outputs = self.output_linear(h)
            outputs = torch.cat([outputs[..., :3], F.softplus(outputs[..., 3:], beta=10)], -1)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

class NeRF_camlatent_layer(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_cam=0, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF_camlatent_layer, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_cam = input_ch_cam
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.ch_cam_linear = nn.Linear(input_ch_cam, input_ch_cam)

        self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + input_ch_cam + W, W//2, activation="relu")])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    def forward(self, x):
        input_pts, input_views, input_cam = torch.split(x, [self.input_ch, self.input_ch_views, self.input_ch_cam], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)

            input_cam = self.ch_cam_linear(input_cam)
            input_cam = F.relu(input_cam)

            h = torch.cat([feature, input_views, input_cam], -1)
            
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, F.softplus(alpha, beta=10)], -1)
        else:
            outputs = self.output_linear(h)
            outputs = torch.cat([outputs[..., :3], F.softplus(outputs[..., 3:], beta=10)], -1)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


class NeRF_camlatent_add(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_cam=0, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF_camlatent_add, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_cam = input_ch_cam
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.ch_cam_linear = nn.Linear(input_ch_cam, W//2)

        self.views_linears = nn.Linear(input_ch_views + W, W//2)

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")

    def forward(self, x):
        input_pts, input_views, input_cam = torch.split(x, [self.input_ch, self.input_ch_views, self.input_ch_cam], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)

            h1 = torch.cat([feature, input_views], -1)
            h1 = self.views_linears(h1)
            h1 = F.relu(h1)

            h2 = self.ch_cam_linear(input_cam)             
            h2 = F.relu(h2)
            
            ## Add
            h = h1 + h2

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, F.softplus(alpha, beta=10)], -1)
        else:
            outputs = self.output_linear(h)
            outputs = torch.cat([outputs[..., :3], F.softplus(outputs[..., 3:], beta=10)], -1)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


def select_coordinates(coords, N_rand):
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    return select_coords

def get_ray_dirs(H, W, intrinsic, c2w, coords=None):
    device = intrinsic.device
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    if coords is None:
        i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
    else:
        i, j = coords[:, 1], coords[:, 0]
    # conversion from pixels to camera coordinates
    dirs = torch.stack([((i + 0.5)-cx)/fx, (H - (j + 0.5) - cy)/fy, -torch.ones_like(i)], -1) # center of pixel
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    return rays_d

# Ray helpers
def get_rays(H, W, intrinsic, c2w, coords=None):
    rays_d = get_ray_dirs(H, W, intrinsic, c2w, coords)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def sample_pdf_return_u(bins, weights, N_samples, det=False, pytest=False, load_u=None):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    if load_u is None:
        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

    ## Otherwise, take the saved u
    else:
        u = load_u

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples, u


def sample_pdf_joint(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(cdf.shape[0], 1)
    else:
        ## Joint samples
        u = torch.rand(N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(cdf.shape[0], 1)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

def sample_pdf_joint_return_u(bins, weights, N_samples, det=False, pytest=False, load_u=None):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if load_u is None:
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
            u = u.unsqueeze(0).repeat(cdf.shape[0], 1)
        else:
            ## Joint samples
            u = torch.rand(N_samples, device=bins.device)
            u = u.unsqueeze(0).repeat(cdf.shape[0], 1)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)
    else:
        u = load_u

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples, u

def pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u_diff):
    
    ln_term = torch.log(T_left) - torch.log(T_left - u_diff)
    discriminant = tau_left**2 + torch.div( 2 * (tau_right - tau_left) * ln_term , s_right - s_left)
    t = torch.div( (s_right - s_left) * (-tau_left + torch.sqrt(torch.max(torch.zeros_like(discriminant), discriminant))) , tau_right - tau_left)

    # print("Inside increasing case")
    # print(torch.isnan(ln_term).any())
    # print(torch.sum(discriminant<0))
    # print(discriminant[discriminant<0])
    # print(torch.isnan(t).any())
    # print("---------")

    sample = s_left + t

    return sample


def pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u_diff):
    
    ln_term = torch.log(T_left) - torch.log(T_left - u_diff)
    discriminant = tau_left**2 - torch.div( 2 * (tau_left - tau_right) * ln_term , s_right - s_left)
    t = torch.div( (s_right - s_left) * (tau_left - torch.sqrt(torch.max(torch.zeros_like(discriminant), discriminant))) , tau_left - tau_right)
    sample = s_left + t

    return sample

def pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ### Fix this, need negative sign
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 + torch.div( 2 * (tau_right - tau_left) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )
    t = torch.div( (s_right - s_left) * (-tau_left + torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_left - tau_right))

    ### Printing to debug ###
    print("====Increasing case=====")
    print("ln term")
    print(ln_term)
    print("(1-u)/T_left")
    print(torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ))
    print()
    print("discriminant")
    print(discriminant)
    print("s_right - s_left")
    print(torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left))
    print("div term in discriminant")
    print(torch.div( 2 * (tau_right - tau_left) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) ))
    print()
    print("t")
    print(t)
    print("denominator: tau_left - tau_right")
    print(torch.max(torch.ones_like(tau_left)*epsilon, tau_left - tau_right))
    print("================")
    #########################

    sample = s_left + t

    return sample


def pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ### Fix this, need negative sign
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 - torch.div( 2 * (tau_left - tau_right) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )
    t = torch.div( (s_right - s_left) * (tau_left - torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_left - tau_right))
    sample = s_left + t

    ### Printing to debug ###
    print("====Decreasing case=====")
    print("ln term")
    print(ln_term)
    print("(1-u)/T_left")
    print(torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ))
    print()
    print("discriminant")
    print(discriminant)
    print("s_right - s_left")
    print(torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left))
    print("div term in discriminant")
    print(torch.div( 2 * (tau_right - tau_left) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) ))
    print()
    print("t")
    print(t)
    print("denominator: tau_left - tau_right")
    print(torch.max(torch.ones_like(tau_left)*epsilon, tau_left - tau_right))
    print("================")
    #########################

    return sample

def sample_pdf_reformulation(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False, quad_solution_v2=False, zero_threshold = 1e-4, epsilon_=1e-3):
    
    ### This needs to be fixed...
    ### Get pdf
    # print("In compute PDF")
    # print(weights.shape)
    # print(bins.shape)

    ### bins = z_vals, ie bin boundaries, input does not include near and far plane yet ## N_samples, with near and far it will become N_samples+2
    ### weights is the PMF of each bin ## N_samples + 1

    bins = torch.cat([near, bins, far], -1)
    
    ### Debug that it will integrate to 1, we made it a way that the far plane is always opaque
    # curr_sum = torch.sum(weights, axis=-1)
    # print(curr_sum)
    # exit()
    
    pdf = weights # make into a probability distribution

    # print("PDF of a ray")
    # print(pdf)
    # print(pdf.shape)
    # exit()

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # print("Computed for CDF")
    # print(cdf)
    # print(cdf.shape)
    # exit()

    ### Overwrite to always have a cdf to end in 1.0 --> I checked and it doesn't always integrate to 1..., make tau at far plane larger?
    cdf[:,-1] = 1.0

    # print("Current shapes")
    # print(near)
    # print(far)
    # print(cdf.shape)
    # print(bins.shape)
    # print(tau.shape)
    # print(T.shape)
    # print()
    # # print(tau_diff.shape)
    # # # print(cdf.shape)
    # # # print(torch.min(cdf))
    # # # print(torch.max(cdf))
    # # # exit()
    # # print()
    # exit()

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)

    ### Get tau diffs, this is to split the case between constant (left and right bin are equal), increasing and decreasing
    tau_diff = tau[...,1:] - tau[...,:-1]
    matched_shape_tau = [inds_g.shape[0], inds_g.shape[1], tau_diff.shape[-1]]
    
    # print("Debugging tau_diff")
    # print("matched tau shape")
    # print(matched_shape_tau)
    # print("tau shape")
    # print(tau.shape)
    # print("cdf shape")
    # print(cdf.shape)
    # print("below shape")
    # print(below.shape)

    tau_diff_g = torch.gather(tau_diff.unsqueeze(1).expand(matched_shape_tau), 2, below.unsqueeze(-1)).squeeze()
    # print("tau diff shape")
    # print(tau_diff_g.shape)

    # print("=====================")
    # print("CDF")
    # print(cdf)
    # print()
    # print(below)
    # exit()

    s_left = bins_g[...,0]
    s_right = bins_g[...,1]
    T_left = T_g[...,0]
    tau_left = tau_g[...,0]
    tau_right = tau_g[...,1]


    print("=======Begin iteration=========")
    print("tau_left - tau_right")
    print(tau_left - tau_right)
    print("tau_diff_g")
    print(tau_diff_g)


    # ### Debug
    # print("Importance sampling")
    # # print(tau_diff_g)
    # # print(tau_diff_g.shape)
    # print(s_left.shape)
    # print(s_right.shape)
    # print(T_left.shape)
    # print(tau_left.shape)
    # print(tau_right.shape)
    # # print()
    # exit()
    # ####

    # zero_threshold = 1e-4

    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)
    # print("Number of constant cases")
    # print(torch.sum(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold)))
    # print()

    if not quad_solution_v2:
        ### Increasing
        samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples1)
        # samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u), samples1)
        # print("Number of increasing cases")
        # print(torch.sum(tau_diff_g > zero_threshold))
        # print()

        ### Decreasing
        samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples2)
        # samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u), samples2)
        # print("Number of decreasing cases")
        # print(torch.sum(tau_diff_g < -zero_threshold))

    else:
        ### Increasing
        samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples1)
        # print("Number of increasing cases")
        # print(torch.sum(tau_diff_g > zero_threshold))
        # print()

        ### Decreasing
        samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples2)
        # print("Number of decreasing cases")
        # print(torch.sum(tau_diff_g < -zero_threshold))

    print("Samples 1")
    print(samples1)
    print("Samples 2")
    print(samples2)
    print("Samples 3")
    print(samples3)
    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)

    print("Samples")
    print(samples)

    # print(samples)
    # print("Does nan exist in samples selected")
    # print(torch.isnan(samples3).any())
    # print(torch.isnan(samples).any())

    ###################################
    ############## TODO ###############
    ###################################
    ### Also return these for custom autograd
    ### T_below, tau_below, bin_below
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]
    ###################################


    return samples, T_below, tau_below, bin_below

def sample_pdf_reformulation_return_u(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False, load_u=None, quad_solution_v2=False, zero_threshold = 1e-4, epsilon=1e-3):
    

    bins = torch.cat([near, bins, far], -1)   
    pdf = weights # make into a probability distribution

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    ### Overwrite to always have a cdf to end in 1.0 --> I checked and it doesn't always integrate to 1..., make tau at far plane larger?
    cdf[:,-1] = 1.0

    if load_u is None:
        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)
    else:
        u = load_u

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)

    ### Get tau diffs, this is to split the case between constant (left and right bin are equal), increasing and decreasing
    tau_diff = tau[...,1:] - tau[...,:-1]
    matched_shape_tau = [inds_g.shape[0], inds_g.shape[1], tau_diff.shape[-1]]
    tau_diff_g = torch.gather(tau_diff.unsqueeze(1).expand(matched_shape_tau), 2, below.unsqueeze(-1)).squeeze()

    s_left = bins_g[...,0]
    s_right = bins_g[...,1]
    T_left = T_g[...,0]
    tau_left = tau_g[...,0]
    tau_right = tau_g[...,1]

    # zero_threshold = 1e-4

    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)

    if not quad_solution_v2:
        ### Increasing
        samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples1)

        ### Decreasing
        samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples2)

    else:
        ### Increasing
        samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples1)

        ### Decreasing
        samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples2)

    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)

    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]
    ###################################


    return samples, T_below, tau_below, bin_below, u

def sample_pdf_reformulation_joint(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False, quad_solution_v2=False, zero_threshold = 1e-4, epsilon=1e-3):

    bins = torch.cat([near, bins, far], -1)
    
    pdf = weights # make into a probability distribution

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    ### Overwrite to always have a cdf to end in 1.0 --> I checked and it doesn't always integrate to 1..., make tau at far plane larger?
    cdf[:,-1] = 1.0

    ### Joint samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(cdf.shape[0], 1)
    else:
        u = torch.rand(N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(cdf.shape[0], 1)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)

    ### Get tau diffs, this is to split the case between constant (left and right bin are equal), increasing and decreasing
    tau_diff = tau[...,1:] - tau[...,:-1]
    matched_shape_tau = [inds_g.shape[0], inds_g.shape[1], tau_diff.shape[-1]]

    tau_diff_g = torch.gather(tau_diff.unsqueeze(1).expand(matched_shape_tau), 2, below.unsqueeze(-1)).squeeze()

    s_left = bins_g[...,0]
    s_right = bins_g[...,1]
    T_left = T_g[...,0]
    tau_left = tau_g[...,0]
    tau_right = tau_g[...,1]

    # zero_threshold = 1e-4

    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)

    if not quad_solution_v2:
        ### Increasing
        samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples1)

        ### Decreasing
        samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples2)

    else:
        ### Increasing
        samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples1)

        ### Decreasing
        samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples2)

    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)


    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]


    return samples, T_below, tau_below, bin_below

def sample_pdf_reformulation_joint_return_u(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False, load_u=None, quad_solution_v2=False, zero_threshold = 1e-4, epsilon=1e-3):

    bins = torch.cat([near, bins, far], -1)
    
    pdf = weights # make into a probability distribution

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    ### Overwrite to always have a cdf to end in 1.0 --> I checked and it doesn't always integrate to 1..., make tau at far plane larger?
    cdf[:,-1] = 1.0

    if load_u is None:
        ### Joint samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
            u = u.unsqueeze(0).repeat(cdf.shape[0], 1)
        else:
            u = torch.rand(N_samples, device=bins.device)
            u = u.unsqueeze(0).repeat(cdf.shape[0], 1)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            new_shape = list(cdf.shape[:-1]) + [N_samples]
            if det:
                u = np.linspace(0., 1., N_samples)
                u = np.broadcast_to(u, new_shape)
            else:
                u = np.random.rand(*new_shape)
            u = torch.Tensor(u)

    else:
        u = load_u

    # Invert CDF
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)


    ### Get tau diffs, this is to split the case between constant (left and right bin are equal), increasing and decreasing
    tau_diff = tau[...,1:] - tau[...,:-1]
    matched_shape_tau = [inds_g.shape[0], inds_g.shape[1], tau_diff.shape[-1]]
    tau_diff_g = torch.gather(tau_diff.unsqueeze(1).expand(matched_shape_tau), 2, below.unsqueeze(-1)).squeeze()

    s_left = bins_g[...,0]
    s_right = bins_g[...,1]
    T_left = T_g[...,0]
    tau_left = tau_g[...,0]
    tau_right = tau_g[...,1]

    # zero_threshold = 1e-4

    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)

    if not quad_solution_v2:
        ### Increasing
        samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples1)

        ### Decreasing
        samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples2)

    else:
        ### Increasing
        samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples1)

        ### Decreasing
        samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples2)

    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)


    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]


    return samples, T_below, tau_below, bin_below, u


class Scale_Gradient_PDF(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, samples, T_below, tau_below, bin_below, samples_raw):
        ctx.save_for_backward(samples, T_below, tau_below, bin_below, samples_raw)

        return samples

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_samples):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        samples, T_below, tau_below, bin_below, samples_raw = ctx.saved_tensors
        grad_T_below = grad_tau_below = grad_bin_below = grad_samples_raw = None


        #### To evaluate tau(s) --> from samples_raw (make sure to use no_grad here) transform (currently: ReLU)
        tau_samples = samples_raw[...,3]
        tau_samples = F.relu(tau_samples)

        ### Based on derivation
        # print("In custom backward.")
        # print(T_below.shape)
        # print(tau_samples.shape)
        # print(tau_below.shape)
        # print(samples.shape)
        # print(bin_below.shape)
        # exit()

        # print(T_below*tau_samples)
        # print(-0.5*(tau_samples+tau_below)(samples-bin_below))
        # print(torch.exp(-0.5*(tau_samples+tau_below)(samples-bin_below)))

        f_s = T_below*tau_samples*torch.exp(-0.5*(tau_samples+tau_below)*(samples-bin_below))
        grad_scale = 1./torch.max(f_s, torch.ones_like(f_s, device=f_s.device)*1e-3) ### prevent nan
        # grad_scale = 1./torch.max(f_s, torch.ones_like(f_s, device=f_s.device)*1e-2) ### prevent nan

        # print("=========")
        # print(T_below)
        # print(tau_samples)
        # print(torch.exp(-0.5*(tau_samples+tau_below)*(samples-bin_below)))
        # print()

        # print("f_s")
        # print(f_s.shape)
        # print(torch.mean(f_s))
        # print(torch.min(f_s))
        # print("grad_scale")
        # print(grad_scale.shape)
        # print(torch.mean(grad_scale))
        # print(torch.max(grad_scale))
        # print()

        # # These needs_input_grad checks are optional and there only to
        # # improve efficiency. If you want to make your code simpler, you can
        # # skip them. Returning gradients for inputs that don't require it is
        # # not an error.
        # if ctx.needs_input_grad[0]:
        #     grad_input = grad_output.mm(weight)
        # if ctx.needs_input_grad[1]:
        #     grad_weight = grad_output.t().mm(input)
        # if bias is not None and ctx.needs_input_grad[2]:
        #     grad_bias = grad_output.sum(0)

        scaled_grad_samples = grad_scale * grad_samples

        ### use negative
        # scaled_grad_samples = -grad_scale * grad_samples

        return scaled_grad_samples, grad_T_below, grad_tau_below, grad_bin_below, grad_samples_raw


def sample_pdf_scratch(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)

    # print(bins.shape)
    # print(weights.shape)
    # print()
    # print(pdf.shape)

    cdf = torch.cumsum(pdf, -1)
    # print(cdf.shape)
    
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))


    # print(cdf.shape)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        # print("Sampling random u's ")
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

        # print(list(cdf.shape[:-1]) + [N_samples])
        # print(u.shape)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    # print(u.shape)

    inds = torch.searchsorted(cdf, u, right=True)

    # print(inds.shape)
    # print(inds)

    below = torch.max(torch.zeros_like(inds-1), inds-1)
    # print(below)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    # print(above)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)
    
    # print(inds_g.shape)
    # print(inds_g)
    
    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    # exit()

    return samples


























