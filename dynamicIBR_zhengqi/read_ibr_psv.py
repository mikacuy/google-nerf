"""Functions for reading dynamic IBR data."""

import collections

from absl import app
from etils import epath
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

INPUT_PATH = '/cns/jn-d/home/parallax-dev/zhengqili/DynamicIBR/monocular-video-1/F_165554287/PSVs'
NPZ_NAME = '00010'


def raw2outputs(input_data, gamma):
  """Converts raw outputs to readable formats."""
  alpha_dy = input_data['alpha_dy']
  alpha_static = input_data['alpha_static']

  rgb_dy = input_data['color_dy']
  rgb_static = input_data['color_static']

  z_vals = input_data['z_vals']

  # how many points we consider for the margin for surface of dynamic object
  num_pts = 2
  margin = (np.max(z_vals) - np.min(z_vals)) / 64. * num_pts

  # Eq. (3): T
  alpha = 1 - (1 - alpha_static) * (1 - alpha_dy)

  t = np.cumprod(1. - alpha + 1e-10, axis=-1)[..., :-1]  # [N_rays, N_samples-1]

  t = np.concatenate((np.ones_like(t[..., 0:1]), t),
                     axis=-1)  # [N_rays, N_samples]

  # FIRST PASS
  # Force single surface of dynamic object weighted by accumulated alpha
  weights_dy = alpha_dy * t     # [N_rays, N_samples]
  depth_map_dy = np.sum(weights_dy * z_vals, axis=-1, keepdims=True)

  sparse_w = np.exp(-(gamma*(weights_dy)**2))
  for i in range(z_vals.shape[-1]):
    front_mask = z_vals[..., i] > (depth_map_dy[..., 0] + margin
                                  )  # cut density behind surface only
    sparse_w[..., i] *= np.float32(front_mask)

  sparse_w = 1. - sparse_w

  # SECOND PASS
  alpha_dy = alpha_dy * sparse_w
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

  return collections.OrderedDict([('rgb', rgb_map),
                                  ('rgb_static', rgb_map_static),
                                  ('rgb_dy', rgb_map_dy), ('depth', depth_map),
                                  ('weights_dy', weights_dy),
                                  ('weights_st', weights_static),
                                  ('alpha_dy', alpha_dy),
                                  ('alpha', alpha), ('weights', weights),
                                  ('z_vals', z_vals)])


def main(_):
  crop_ratio = 0.03
  gamma = 50  # determinie where we start to clamp density

  root = epath.Path(INPUT_PATH)
  camera_path = root / (NPZ_NAME + '_camera.npy')
  with camera_path.open('rb') as f:
    camera = np.load(f)[0]

  h, w = camera[0], camera[1]
  # intrinsic = camera[2:2 + 16].reshape(4, 4)
  # cam_c2w = camera[2 + 16:2 + 32].reshape(4, 4)
  data = {}

  def load_into_data(channel_name):
    path = root / (NPZ_NAME + '_' + channel_name + '.npy')
    with path.open('rb') as f:
      data[channel_name] = np.load(f)

  load_into_data('alpha_dy')
  load_into_data('alpha_static')
  load_into_data('color_dy')
  load_into_data('color_static')
  load_into_data('z_vals')

  ret = raw2outputs(data, gamma)

  full_rgb = ret['rgb']  # full rendered image
  st_rgb = ret['rgb_static']  # static component
  dy_rgb = ret['rgb_dy']  # dynamic component
  disparity = 1. / ret['depth']  # dynamic component

  h, w = full_rgb.shape[:2]
  crop_h = int(h * crop_ratio)
  crop_w = int(w * crop_ratio)
  full_rgb = np.clip(full_rgb[crop_h:h - crop_h, crop_w:w - crop_w, ...], 0.,
                     1.)
  st_rgb = np.clip(st_rgb[crop_h:h - crop_h, crop_w:w - crop_w, ...], 0., 1.)
  dy_rgb = np.clip(dy_rgb[crop_h:h - crop_h, crop_w:w - crop_w, ...], 0., 1.)
  disparity = disparity[crop_h:h - crop_h, crop_w:w - crop_w, ...]

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


if __name__ == '__main__':
  app.run(main)
