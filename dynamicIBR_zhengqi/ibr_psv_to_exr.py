"""Functions for reading dynamic IBR data.

In NUKE, use this script to create the merge nodes to test the composition:

connect_a = nuke.toNode('Read1')
connect_b = connect_a
for i in range(63, 0, -1):
  m = nuke.nodes.Merge(name=f'layer{i:02}')
  m.setInput(1, connect_a)
  m.setInput(0, connect_b)
  m.knob('A').setValue(f'layer{(i-1):02}')
  if i == 63:
    m.knob('B').setValue(f'layer{(i):02}')
  connect_b = m


To create MPI geometry, use this script:

z_vals = [...]

read_node = nuke.toNode('Read1')
scene_node = nuke.nodes.Scene(name='mpi')
for i in range(64):
  s = nuke.nodes.Shuffle(name=f'shuffle{i:02}')
  s.setInput(0, read_node)
  s.knob('in').setValue(f'layer{i:02}')
  c = nuke.nodes.Card(name=f'card{i:02}')
  c.setInput(0, s)
  c.knob('translate').setValue([0,0,-z_vals[i]])
  c.knob('uniform_scale').setValue(z_vals[i])
  scene_node.setInput(i, c)
"""

import collections

from absl import app
from etils import epath
import numpy as np
from google3.pyglib.concurrent import parallel
from google3.research.vision.viscam.image_utils.python import exr_to_numpy

INPUT_PATH = '/tmp/monocular-video-1/F_165554287/PSVs'


def raw2outputs(input_data):
  """Converts raw outputs to readable formats."""
  alpha_dy = input_data['alpha_dy']
  alpha_static = input_data['alpha_static']

  rgb_dy = input_data['color_dy']
  rgb_static = input_data['color_static']

  z_vals = input_data['z_vals']
  # Eq. (3): T
  alpha = 1 - (1 - alpha_static) * (1 - alpha_dy)

  t = np.cumprod(1. - alpha + 1e-10, axis=-1)[..., :-1]  # [N_rays, N_samples-1]

  t = np.concatenate((np.ones_like(t[..., 0:1]), t),
                     axis=-1)  # [N_rays, N_samples]

  # maths show weights, and summation of weights along a ray,
  # are always inside [0, 1]
  weights_dy = alpha_dy * t  # [N_rays, N_samples]

  rgb_map_dy = np.sum(weights_dy[..., None] * rgb_dy, axis=-2)  # [N_rays, 3]

  weights_static = alpha_static * t  # [N_rays, N_samples]
  rgb_map_static = np.sum(
      weights_static[..., None] * rgb_static, axis=-2)  # [N_rays, 3]

  rgb_map = rgb_map_dy + rgb_map_static

  weights = alpha * t  # (N_rays, N_samples_)

  rgb_comp = rgb_dy * weights_dy[..., np.newaxis] + rgb_static * (
      weights_static[..., np.newaxis])

  depth_map = np.sum(weights * z_vals, axis=-1)  # [N_rays,]

  rgba_comp = np.concatenate((rgb_comp, weights[..., np.newaxis]), axis=-1)

  z_vals = z_vals[0, 0, :]

  return collections.OrderedDict([('rgb', rgb_map), ('rgba_comp', rgba_comp),
                                  ('depth', depth_map), ('z_vals', z_vals)])


def export_frame(npz_name):
  """Exports a single frame to .exr."""
  root = epath.Path(INPUT_PATH)
  camera_path = root / (npz_name + '_camera.npy')
  with camera_path.open('rb') as f:
    camera = np.load(f)[0]

  # h, w = camera[0], camera[1]
  camera_intrinsics = camera[2:2 + 16].reshape(4, 4)[:3, :3]
  camera_to_world = camera[2 + 16:2 + 32].reshape(4, 4)
  world_to_camera = np.linalg.inv(camera_to_world)
  data = {}

  def load_into_data(channel_name):
    print(f'Loading {channel_name}')
    path = root / (npz_name + '_' + channel_name + '.npy')
    with path.open('rb') as f:
      data[channel_name] = np.load(f)

  load_into_data('alpha_dy')
  load_into_data('alpha_static')
  load_into_data('color_dy')
  load_into_data('color_static')
  load_into_data('z_vals')

  ret = raw2outputs(data)

  full_rgb = ret['rgb']  # full rendered image
  rgba_comp = ret['rgba_comp']  # Per-layer rgba
  disparity = 1. / ret['depth']  # dynamic component

  rgba_comp = np.reshape(rgba_comp,
                         (rgba_comp.shape[0], rgba_comp.shape[1], -1))

  output_layers = np.concatenate(
      (full_rgb, disparity[..., np.newaxis], rgba_comp), axis=-1)
  output_layer_names = ['R', 'G', 'B', 'Z']
  # pylint: disable=g-complex-comprehension
  output_layer_names += [
      f'layer{i:02}.{x}' for i in range(64) for x in ['r', 'g', 'b', 'a']
  ]
  print(output_layer_names)

  output_path = epath.Path('/tmp') / (npz_name + '.exr')
  print(output_path)
  exr_to_numpy.write_exr(
      output_path, output_layers.astype(np.float16), output_layer_names, {
          'worldToCamera': world_to_camera,
          'cameraToWorld': camera_to_world,
          'cameraIntrinsics': camera_intrinsics,
          'mpi_z_0': np.reshape(ret['z_vals'][:16], (4, 4)),
          'mpi_z_1': np.reshape(ret['z_vals'][16:32], (4, 4)),
          'mpi_z_2': np.reshape(ret['z_vals'][32:48], (4, 4)),
          'mpi_z_3': np.reshape(ret['z_vals'][48:], (4, 4)),
      })


def main(_):
  frame_kwargs = [{'npz_name': f'{i:05}'} for i in range(5, 6)]
  parallel.RunInParallel(
      function=export_frame,
      list_of_kwargs_to_function=frame_kwargs,
      num_workers=len(frame_kwargs))

if __name__ == '__main__':
  app.run(main)
