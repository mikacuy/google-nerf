import numpy as np
import torch
import collections


Rays = collections.namedtuple('Rays', ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device),
                     cdf,
                     torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)],
                    dim=-1)

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = u + u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, _ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, _ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples

####### For Piecewise Linear #######
def pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ### Fix this, need negative sign
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 + torch.div( 2 * (tau_right - tau_left) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )

    t = torch.div( (s_right - s_left) * (-tau_left + torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_right - tau_left))

    ### clamp t to [0, s_right - s_left]
    # print("t clamping")
    # print(torch.max(t))
    t = torch.clamp(t, torch.ones_like(t, device=t.device)*epsilon, s_right - s_left)
    # print(torch.max(t))
    # print()

    sample = s_left + t

    return sample


def pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=1e-3):
    ### Fix this, need negative sign
    ln_term = -torch.log(torch.max(torch.ones_like(T_left)*epsilon, torch.div(1-u, torch.max(torch.ones_like(T_left)*epsilon,T_left) ) ))
    discriminant = tau_left**2 - torch.div( 2 * (tau_left - tau_right) * ln_term , torch.max(torch.ones_like(s_right)*epsilon, s_right - s_left) )
    t = torch.div( (s_right - s_left) * (tau_left - torch.sqrt(torch.max(torch.ones_like(discriminant)*epsilon, discriminant))) , torch.max(torch.ones_like(tau_left)*epsilon, tau_left - tau_right))

    ### clamp t to [0, s_right - s_left]
    # print("t clamping")
    # print(torch.max(t))
    t = torch.clamp(t, torch.ones_like(t, device=t.device)*epsilon, s_right - s_left)
    # print(torch.max(t))
    # print()

    sample = s_left + t

    return sample

def sample_pdf_reformulation(bins, weights, tau, T, near, far, N_samples, det=False, pytest=False, zero_threshold = 1e-4, epsilon_=1e-3):

    # print(bins.shape)
    # exit()

    bins = torch.cat([near, bins, far], -1)
    
    pdf = weights 
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


    dummy = torch.ones(s_left.shape, device=s_left.device)*-1.0

    ### Constant interval, take the left bin
    samples1 = torch.where(torch.logical_and(tau_diff_g < zero_threshold, tau_diff_g > -zero_threshold), s_left, dummy)

    ### Increasing
    samples2 = torch.where(tau_diff_g >= zero_threshold, pw_linear_sample_increasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples1)
    # print("Number of increasing cases")
    # print(torch.sum(tau_diff_g > zero_threshold))
    # print()

    ### Decreasing
    samples3 = torch.where(tau_diff_g <= -zero_threshold, pw_linear_sample_decreasing_v2(s_left, s_right, T_left, tau_left, tau_right, u, epsilon=epsilon_), samples2)
    # print("Number of decreasing cases")
    # print(torch.sum(tau_diff_g < -zero_threshold))
    # print()
    ## Check for nan --> need to figure out why
    samples = torch.where(torch.isnan(samples3), s_left, samples3)


    tau_g = torch.gather(tau.unsqueeze(1).expand(matched_shape), 2, inds_g)
    T_g = torch.gather(T.unsqueeze(1).expand(matched_shape), 2, inds_g)

    T_below = T_g[...,0]
    tau_below = tau_g[...,0]
    bin_below = bins_g[...,0]

    return samples, T_below, tau_below, bin_below

####################################

def convert_to_ndc(origins, directions, focal, w, h, near=1.):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane
    t = -(near + origins[..., 2]) / (directions[..., 2] + 1e-15)
    origins = origins + t[..., None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / (oz + 1e-15))
    o1 = -((2 * focal) / h) * (oy / (oz+ 1e-15) )
    o2 = 1 + 2 * near / (oz+ 1e-15)

    d0 = -((2 * focal) / w) * (dx / (dz+ 1e-15) - ox / (oz+ 1e-15))
    d1 = -((2 * focal) / h) * (dy / (dz+ 1e-15) - oy / (oz+ 1e-15))
    d2 = -2 * near / (oz+ 1e-15)

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = torch.sum(d ** 2, dim=-1, keepdim=True) + 1e-10

    if diag:
        d_outer_diag = d ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.

    Args:
    d: torch.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

    Returns:
    a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                          (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                  (hw**4) / (3 * mu**2 + hw**2))
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.

    Args:
      d: torch.float32 3-vector, the axis of the cylinder
      t0: float, the starting distance of the cylinder.
      t1: float, the ending distance of the cylinder.
      radius: float, the radius of the cylinder
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

    Returns:
      a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

    Args:
      t_vals: float array, the "fencepost" distances along the ray.
      origins: float array, the ray origin coordinates.
      directions: float array, the ray direction vectors.
      radii: float array, the radii (base radii for cones) of the rays.
      diag: boolean, whether or not the covariance matrices should be diagonal.

    Returns:
      a tuple of arrays of means and covariances.
    """
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, lindisp, ray_shape):
    """Stratified sampling along the rays.

    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      num_samples: int.
      near: torch.tensor, [batch_size, 1], near clip.
      far: torch.tensor, [batch_size, 1], far clip.
      randomized: bool, use randomized stratified sampling.
      lindisp: bool, sampling linearly in disparity rather than depth.

    Returns:
      t_vals: torch.tensor, [batch_size, num_samples], sampled z values.
      means: torch.tensor, [batch_size, num_samples, 3], sampled means.
      covs: torch.tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0., 1., num_samples + 1,  device=origins.device)
    if lindisp:
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        t_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples + 1])
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)

    return t_vals, (means, covs)


def resample_along_rays(origins, directions, radii, t_vals, weights, randomized, stop_grad, resample_padding, ray_shape):
    """Resampling.

    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      weights: torch.tensor(float32), weights for t_vals
      randomized: bool, use randomized samples.
      stop_grad: bool, whether or not to backprop through sampling.
      resample_padding: float, added to the weights before normalizing.

    Returns:
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      points: torch.tensor(float32), [batch_size, num_samples, 3].
    """
    if stop_grad:
        with torch.no_grad():
            weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

            # Add in a constant (the sampling function will renormalize the PDF).
            weights = weights_blur + resample_padding

            new_t_vals = sorted_piecewise_constant_pdf(
                t_vals,
                weights,
                t_vals.shape[-1],
                randomized,
            )
    else:
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
            t_vals,
            weights,
            t_vals.shape[-1],
            randomized,
        )
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)


def resample_along_rays_piecewise_linear(origins, directions, radii, t_vals, weights, randomized, stop_grad, resample_padding, ray_shape, tau, T, near, far):
    """Resampling.

    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      weights: torch.tensor(float32), weights for t_vals
      randomized: bool, use randomized samples.
      stop_grad: bool, whether or not to backprop through sampling.
      resample_padding: float, added to the weights before normalizing.

    Returns:
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      points: torch.tensor(float32), [batch_size, num_samples, 3].
    """
    if stop_grad:
        with torch.no_grad():
            # weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
            # weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            # weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

            # Add in a constant (the sampling function will renormalize the PDF).
            # weights = weights_blur + resample_padding

            z_vals = 0.5 * (t_vals[...,:-1] + t_vals[...,1:])
            new_t_vals, _, _, _ = sample_pdf_reformulation(z_vals, weights, tau, T, near, far, t_vals.shape[-1], det= (not randomized), pytest=False)
            new_t_vals = torch.clamp(new_t_vals, near, far)
            new_t_vals, _ = torch.sort(new_t_vals, -1)

            # print("In resample_along_rays_piecewise_linear")
            # print(z_vals)

            # dists = z_vals[...,1:] - z_vals[...,:-1]
            # print(dists)
            # print((dists>=0).all())
            # print()

            # new_t_vals = sorted_piecewise_constant_pdf(
            #     t_vals,
            #     weights,
            #     t_vals.shape[-1],
            #     randomized,
            # )

            
    else:
        # weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        # weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        # weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # # Add in a constant (the sampling function will renormalize the PDF).
        # weights = weights_blur + resample_padding

        z_vals = 0.5 * (t_vals[...,:-1] + t_vals[...,1:])
        new_t_vals, _, _, _ = sample_pdf_reformulation(z_vals, weights, tau, T, near, far, t_vals.shape[-1], det= (not randomized), pytest=False)
        new_t_vals = torch.clamp(new_t_vals, near, far)
        new_t_vals, _ = torch.sort(new_t_vals, -1)

        # new_t_vals = sorted_piecewise_constant_pdf(
        #     t_vals,
        #     weights,
        #     t_vals.shape[-1],
        #     randomized,
        # )


    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)



def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    """Volumetric Rendering Function.

    Args:
    rgb: torch.tensor(float32), color, [batch_size, num_samples, 3]
    density: torch.tensor(float32), density, [batch_size, num_samples, 1].
    t_vals: torch.tensor(float32), [batch_size, num_samples].
    dirs: torch.tensor(float32), [batch_size, 3].
    white_bkgd: bool.

    Returns:
    comp_rgb: torch.tensor(float32), [batch_size, 3].
    disp: torch.tensor(float32), [batch_size].
    acc: torch.tensor(float32), [batch_size].
    weights: torch.tensor(float32), [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)
    distance = (weights * t_mids).sum(dim=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights, alpha


def volumetric_rendering_piecewise_linear(rgb, density, t_vals, dirs, white_bkgd, near, far, color_mode="midpoint"):
    """Volumetric Rendering Function.

    Args:
    rgb: torch.tensor(float32), color, [batch_size, num_samples, 3]
    density: torch.tensor(float32), density, [batch_size, num_samples, 1].
    t_vals: torch.tensor(float32), [batch_size, num_samples].
    dirs: torch.tensor(float32), [batch_size, 3].
    white_bkgd: bool.

    Returns:
    comp_rgb: torch.tensor(float32), [batch_size, 3].
    disp: torch.tensor(float32), [batch_size].
    acc: torch.tensor(float32), [batch_size].
    weights: torch.tensor(float32), [batch_size, num_samples]
    """
    raw2expr = lambda raw, dists: torch.exp(-raw*dists)
    
    ### Midpoints are the bin boundaries
    z_vals = 0.5 * (t_vals[...,:-1] + t_vals[...,1:])
    z_vals = torch.cat([near, z_vals, far], -1)

    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = dists * torch.norm(dirs[...,None,:], dim=-1)
    tau = torch.cat([torch.ones((density.shape[0], 1), device=density.device)*1e-10, density[...,0], torch.ones((density.shape[0], 1), device=density.device)*1e10], -1)

    # tau = F.relu(tau)

    interval_ave_tau = 0.5 * (tau[...,1:] + tau[...,:-1])

    # print(z_vals.shape)
    # print(density.shape)
    # print(tau.shape)
    # print(dirs.shape)
    # print(dists.shape)
    # print(interval_ave_tau.shape)
    # print()
    
    expr = raw2expr(interval_ave_tau, dists)
    
    T = torch.cumprod(torch.cat([torch.ones((expr.shape[0], 1), device=density.device), expr], -1), -1)
    alpha = (1 - expr)
    weights = alpha * T[:, :-1]

    # print()
    # print("z_vals")
    # print(z_vals)
    # print("dists")
    # print((dists>=0).all())
    # print(dists)
    # print("tau")
    # print(torch.isnan(tau).any())
    # print("expr")
    # print(torch.isnan(expr).any())
    # print(expr)
    # print("alpha")
    # print(torch.isnan(alpha).any())    
    # print("T")
    # print(torch.isnan(T).any())
    # print(T)
    # print("weights")
    # print(torch.isnan(weights).any())

    # print(weights.shape)
    # print(tau.shape)
    # print(T.shape)
    # print(t_vals.shape)
    # print(rgb.shape)
    # print(density.shape)
    # exit()

    # if color_mode == "midpoint":

    rgb_concat = torch.cat([rgb[: ,0, :].unsqueeze(1), rgb, rgb[: ,-1, :].unsqueeze(1)], 1)
    rgb_mid = .5 * (rgb_concat[:, 1:, :] + rgb_concat[:, :-1, :])

    comp_rgb = (weights[..., None] * rgb_mid).sum(dim=-2)
    acc = weights.sum(dim=-1)
    distance = (weights * t_vals).sum(dim=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_vals[:, 0], t_vals[:, -1])

    # weights_to_aggregate = weights[..., 1:]
    # comp_rgb = (weights_to_aggregate[..., None] * rgb).sum(dim=-2)
    # acc = weights_to_aggregate.sum(dim=-1)
    # distance = (weights_to_aggregate * t_mids).sum(dim=-1) / acc
    # distance = torch.clamp(torch.nan_to_num(distance), t_vals[:, 0], t_vals[:, -1])

    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])

    # print(comp_rgb.shape)
    # print(distance.shape)
    # print(acc.shape)
    # exit()

    return comp_rgb, distance, acc, weights, alpha, tau, T




