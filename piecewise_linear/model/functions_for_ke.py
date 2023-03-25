### Piecewise constant case
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


####################################
## Piecewise Linear Reformulation ##
####################################

def pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u_diff):
    
    ln_term = torch.log(T_left) - torch.log(T_left - u_diff)
    discriminant = tau_left**2 + torch.div( 2 * (tau_right - tau_left) * ln_term , s_right - s_left)
    t = torch.div( (s_right - s_left) * (-tau_left + torch.sqrt(torch.max(torch.zeros_like(discriminant), discriminant))) , tau_right - tau_left)

    sample = s_left + t

    return sample


def pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u_diff):
    
    ln_term = torch.log(T_left) - torch.log(T_left - u_diff)
    discriminant = tau_left**2 - torch.div( 2 * (tau_left - tau_right) * ln_term , s_right - s_left)
    t = torch.div( (s_right - s_left) * (tau_left - torch.sqrt(torch.max(torch.zeros_like(discriminant), discriminant))) , tau_left - tau_right)
    sample = s_left + t

    return sample

def pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u):
    EPSILON = 1e-3

    ### Fix this, need negative sign
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*EPSILON, torch.div(1-u, torch.max(torch.ones_like(T_left)*EPSILON,T_left) ) ))
    discriminant = tau_left**2 + torch.div( 2 * (tau_right - tau_left) * ln_term , torch.max(torch.ones_like(s_right)*EPSILON, s_right - s_left) )
    t = torch.div( (s_right - s_left) * (-tau_left + torch.sqrt(torch.max(torch.ones_like(discriminant)*EPSILON, discriminant))) , torch.max(torch.ones_like(tau_left)*EPSILON, tau_left - tau_right))

    sample = s_left + t

    return sample


def pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u):
    EPSILON = 1e-3

    ### Fix this, need negative sign
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*EPSILON, torch.div(1-u, torch.max(torch.ones_like(T_left)*EPSILON,T_left) ) ))
    discriminant = tau_left**2 - torch.div( 2 * (tau_left - tau_right) * ln_term , torch.max(torch.ones_like(s_right)*EPSILON, s_right - s_left) )
    t = torch.div( (s_right - s_left) * (tau_left - torch.sqrt(torch.max(torch.ones_like(discriminant)*EPSILON, discriminant))) , torch.max(torch.ones_like(tau_left)*EPSILON, tau_left - tau_right))
    sample = s_left + t

    return sample

def sample_pdf_reformulation(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False):

    ### bins = z_vals, ie bin boundaries, input does not include near and far plane yet ## N_samples, with near and far it will become N_samples+2
    ### weights is the PMF of each bin ## N_samples + 1

    bins = torch.cat([near, bins, far], -1)
    
    pdf = weights # make into a probability distribution

    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    ### Overwrite to always have a cdf to end in 1.0 --> I checked and it doesn't always integrate to 1..., make tau at far plane larger?
    cdf[:,-1] = 1.0


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

    tau_diff_g = torch.gather(tau_diff.unsqueeze(1).expand(matched_shape_tau), 2, below.unsqueeze(-1)).squeeze()


    s_left = bins_g[...,0]
    s_right = bins_g[...,1]
    T_left = T_g[...,0]
    tau_left = tau_g[...,0]
    tau_right = tau_g[...,1]

    zero_threshold = 1e-4

    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)

    ### Increasing
    samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples1)

    ### Decreasing
    samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing(s_left, s_right, T_left, tau_left, tau_right, u-cdf_g[...,0]), samples2)


    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)


    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]
    ###################################


    return samples, T_below, tau_below, bin_below


