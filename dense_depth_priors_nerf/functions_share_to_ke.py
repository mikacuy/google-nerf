########################
### Current version ####
########################
# Marginal version
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
    
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


### Snippet from train script
rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)


z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
if not is_joint:
    z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
else:
    z_samples = sample_pdf_joint(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)

pred_depth_hyp = z_samples ## P_Vol
########################
########################
########################



########################################
### Previous Reformulation --> wrong ###
########################################
def sample_pdf_reformulation(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False):
    
    bins = torch.cat([near, bins, far], -1)  
    curr_sum = torch.sum(weights, axis=-1)
    
    pdf = weights 

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

    ### Also return these for custom autograd
    ### T_below, tau_below, bin_below
    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]

    return samples, T_below, tau_below, bin_below

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


        f_s = T_below*tau_samples*torch.exp(-0.5*(tau_samples+tau_below)*(samples-bin_below))
        grad_scale = 1./torch.max(f_s, torch.ones_like(f_s, device=f_s.device)*1e-3) ### prevent nan

        scaled_grad_samples = grad_scale * grad_samples

        ### use negative
        # scaled_grad_samples = -grad_scale * grad_samples

        return scaled_grad_samples, grad_T_below, grad_tau_below, grad_bin_below, grad_samples_raw



### Snippet from train script
rgb_map, disp_map, acc_map, weights, depth_map, tau, T = raw2outputs(raw, z_vals, near, far, rays_d, raw_noise_std, pytest=pytest)

if not is_joint:
    z_samples, T_below, tau_below, bin_below = sample_pdf_reformulation(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest)
else:
    z_samples, T_below, tau_below, bin_below = sample_pdf_reformulation_joint(z_vals, weights, tau, T, near, far, N_importance, det=(perturb==0.), pytest=pytest)

pts = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None]
samples_raw = network_query_fn(pts, viewdirs, embedded_cam, run_fn)

z_samples = Scale_Gradient_PDF.apply(z_samples, T_below, tau_below, bin_below, samples_raw)
pred_depth_hyp = z_samples

########################################
########################################
########################################








