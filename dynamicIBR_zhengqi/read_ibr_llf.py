"""Functions for reading dynamic IBR data."""

import collections
import sys
from absl import app
from etils import epath
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

INPUT_PATH = '/cns/jn-d/home/parallax-dev/zhengqili/DynamicIBR/monocular-video-1/PSVs_050'

CLAMP_DYNAMIC = True
CLAMP_STATIC = False

def raw2outputs(input_data, gamma, margin_ratio):
  """Converts raw outputs to readable formats."""
  alpha_dy = input_data['alpha_dy']
  alpha_static = input_data['alpha_static']
  alpha_dy[..., -10:] = 0.

  rgb_dy = input_data['color_dy']
  rgb_static = input_data['color_static']
  t_src = input_data['T']

  z_vals = input_data['z_vals']

  # how many points we consider for the margin for surface of dynamic object

  # Eq. (3): T
  alpha = 1 - (1 - alpha_static) * (1 - alpha_dy)

  t = np.cumprod(1. - alpha + 1e-10, axis=-1)[..., :-1]  # [N_rays, N_samples-1]
  t = np.concatenate((np.ones_like(t[..., 0:1]), t),
                     axis=-1)  # [N_rays, N_samples]

  weights_dy = alpha_dy * t  # [N_rays, N_samples]
  depth_map_dy = np.sum(
      weights_dy * z_vals, axis=-1, keepdims=True)  # [N_rays,]

  sparse_w_dy = np.exp(-(gamma * (weights_dy)**2))
  for i in range(z_vals.shape[-1]):
    front_mask = z_vals[..., i] > (depth_map_dy[..., 0] * margin_ratio)
    sparse_w_dy[..., i] *= np.float32(front_mask)

  sparse_w_dy = 1. - sparse_w_dy

  t_st = np.cumprod(
      1. - alpha_static + 1e-10, axis=-1)[..., :-1]  # [N_rays, N_samples-1]
  t_st = np.concatenate((np.ones_like(t_st[..., 0:1]), t_st),
                        axis=-1)  # [N_rays, N_samples]

  weights_st = alpha_static * t_st  # [N_rays, N_samples]
  depth_map_st = np.sum(
      weights_st * z_vals, axis=-1, keepdims=True)  # [N_rays,]

  sparse_w_st = np.exp(-(gamma * (weights_st)**2))
  for i in range(z_vals.shape[-1]):
    front_mask = z_vals[..., i] > (depth_map_st[..., 0] * margin_ratio)
    sparse_w_st[..., i] *= np.float32(front_mask)

  sparse_w_st = 1. - sparse_w_st

  # SECOND PASS
  if CLAMP_DYNAMIC:
    alpha_dy = alpha_dy * sparse_w_dy

  if CLAMP_STATIC:
    alpha_static = alpha_static * sparse_w_st

  alpha = 1 - (1 - alpha_static) * (1 - alpha_dy)
  t = np.cumprod(1. - alpha + 1e-10, axis=-1)[..., :-1]  # [N_rays, N_samples-1]
  t = np.concatenate((np.ones_like(t[..., 0:1]), t),
                     axis=-1)  # [N_rays, N_samples]
  weights_dy = alpha_dy * t  # [N_rays, N_samples]

  depth_map_dy = np.sum(
      weights_dy * z_vals, axis=-1, keepdims=True)  # [N_rays,]
  rgb_map_dy = np.sum(weights_dy[..., None] * rgb_dy, axis=-2)  # [N_rays, 3]

  weights_static = alpha_static * t  # [N_rays, N_samples]
  rgb_map_static = np.sum(
      weights_static[..., None] * rgb_static, axis=-2)  # [N_rays, 3]

  rgb_map = rgb_map_dy + rgb_map_static

  weights = alpha * t  # (N_rays, N_samples_)

  depth_map = np.sum(weights * z_vals, axis=-1)  # [N_rays,]

  return collections.OrderedDict([('rgb', rgb_map), ('rgb_static', rgb_static),
                                  ('rgb_dy', rgb_dy), ('depth', depth_map),
                                  ('alpha_static', alpha_static),
                                  ('alpha_dy', alpha_dy), ('alpha', alpha),
                                  ('weights', weights), ('z_vals', z_vals),
                                  ('t_src', t_src)])


def light_field_fusion(input_data):
  """Fuse MPIs to one light field."""

  z_vals = input_data[0]['z_vals']
  alpha_dy_psv = []
  alpha_st_psv = []
  color_dy_psv = []
  color_st_psv = []

  for i in range(input_data[0]['t_src'].shape[-1]):  # plane sweep
    alpha_dy_list = []
    alpha_st_list = []
    weight_dy_list = []
    weight_st_list = []
    rgb_dy_list = []
    rgb_st_list = []

    for k in range(len(input_data)):  # img idx
      t_src = input_data[k]['t_src'][..., i]
      alpha_dy = input_data[k]['alpha_dy'][..., i]
      alpha_st = input_data[k]['alpha_static'][..., i]

      weight_dy_src = t_src * alpha_dy
      weight_st_src = t_src * alpha_st

      alpha_dy_list.append(alpha_dy)
      alpha_st_list.append(alpha_st)
      weight_dy_list.append(weight_dy_src)
      weight_st_list.append(weight_st_src)

      rgb_dy = input_data[k]['rgb_dy'][..., i, :]
      rgb_static = input_data[k]['rgb_static'][..., i, :]

      rgb_dy_list.append(rgb_dy)
      rgb_st_list.append(rgb_static)

    alpha_dy_list = np.array(alpha_dy_list)
    alpha_st_list = np.array(alpha_st_list)
    weight_dy_list = np.array(weight_dy_list)
    weight_st_list = np.array(weight_st_list)
    rgb_dy_list = np.array(rgb_dy_list)
    rgb_st_list = np.array(rgb_st_list)

    alpha_dy_final = np.sum((alpha_dy_list * weight_dy_list), axis=0) / np.sum(
        weight_dy_list + 1e-9, axis=0)
    alpha_st_final = np.sum((alpha_st_list * weight_st_list), axis=0) / np.sum(
        weight_st_list + 1e-9, axis=0)

    color_dy_final = np.sum(
        (rgb_dy_list * weight_dy_list[..., None]), axis=0) / np.sum(
            weight_dy_list[..., None] + 1e-9, axis=0)
    color_st_final = np.sum(
        (rgb_st_list * weight_st_list[..., None]), axis=0) / np.sum(
            weight_st_list[..., None] + 1e-9, axis=0)

    alpha_dy_psv.append(alpha_dy_final)
    alpha_st_psv.append(alpha_st_final)
    color_dy_psv.append(color_dy_final)
    color_st_psv.append(color_st_final)

  alpha_dy = np.array(alpha_dy_psv).transpose(1, 2, 0)
  alpha_static = np.array(alpha_st_psv).transpose(1, 2, 0)
  rgb_dy = np.array(color_dy_psv).transpose(1, 2, 0, 3)
  rgb_static = np.array(color_st_psv).transpose(1, 2, 0, 3)

  # standard volume rendering
  alpha = 1 - (1 - alpha_static) * (1 - alpha_dy)

  t = np.cumprod(1. - alpha + 1e-10, axis=-1)[..., :-1]  # [N_rays, N_samples-1]

  t = np.concatenate((np.ones_like(t[..., 0:1]), t),
                     axis=-1)  # [N_rays, N_samples]

  weights_dy = alpha_dy * t  # [N_rays, N_samples]

  rgb_map_dy = np.sum(weights_dy[..., None] * rgb_dy, axis=-2)  # [N_rays, 3]

  weights_static = alpha_static * t  # [N_rays, N_samples]
  rgb_map_static = np.sum(
      weights_static[..., None] * rgb_static, axis=-2)  # [N_rays, 3]

  rgb_map = rgb_map_dy + rgb_map_static

  weights = alpha * t  # (N_rays, N_samples_)

  depth_map = np.sum(weights * z_vals, axis=-1)  # [N_rays,]

  return collections.OrderedDict([('rgb_map_static', rgb_map_static),
                                  ('rgb_map_dy', rgb_map_dy),
                                  ('rgb_map', rgb_map),
                                  ('depth_map', depth_map),
                                  ('alpha_dy', alpha_dy),
                                  ('alpha_static', alpha_static),
                                  ('alpha', alpha), ('rgb_dy', rgb_dy),
                                  ('rgb_static', rgb_static),
                                  ('weights', weights)])


def main(_):
  crop_ratio = 0.03
  gamma = 50  # determinie we start to clamp density from accumulated w
  margin_ratio = 1.15  # determinie where we start to clamp behind the surface

  root = epath.Path(INPUT_PATH)

  def load_into_data(data, npz_name, channel_name):
    path = root / (npz_name + '_' + channel_name + '.npy')
    with path.open('rb') as f:
      data[channel_name] = np.load(f)

  mpis = []
  # sweep each camera index
  for i in range(0, 6):
    print('READ ', i)
    data = {}
    npz_name = '%05d' % i
    load_into_data(data, npz_name, 'alpha_dy')
    load_into_data(data, npz_name, 'alpha_static')
    load_into_data(data, npz_name, 'color_dy')
    load_into_data(data, npz_name, 'color_static')
    load_into_data(data, npz_name, 'z_vals')
    load_into_data(data, npz_name, 'T')
    ret = raw2outputs(data, gamma, margin_ratio)
    mpis.append(ret)

  ret_full = light_field_fusion(mpis)

  for i in range(ret_full['alpha'].shape[-1]):
    final_color = np.float32(ret_full['rgb_static'][..., i, :] *
                             ret_full['alpha_static'][..., i, None])
    final_color += np.float32(ret_full['rgb_dy'][..., i, :] *
                              ret_full['alpha_dy'][..., i, None])
    imageio.imwrite('/tmp/psv_rgba/%03d.png' % i, final_color)

  full_rgb = ret_full['rgb_map']  # full rendered image
  st_rgb = ret_full['rgb_map_static']  # static component
  dy_rgb = ret_full['rgb_map_dy']  # dynamic component
  disparity = 1. / ret_full['depth_map']  # dynamic component

  plt.figure(figsize=(19, 10))
  plt.subplot(2, 2, 1)
  plt.imshow(full_rgb)
  plt.subplot(2, 2, 2)
  plt.imshow(dy_rgb)
  plt.subplot(2, 2, 3)
  plt.imshow(st_rgb)
  plt.subplot(2, 2, 4)
  plt.imshow(disparity, cmap='jet')
  plt.tight_layout()
  plt.savefig('/tmp/test_psv.png')
  print("DONE!")

if __name__ == '__main__':
  app.run(main)
