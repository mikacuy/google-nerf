import os

import cv2
import imageio
import numpy as np
import torch
import json

BLENDER2OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    # 草，这个地方源代码没有乘这个blender2opencv，做这个操作相当于把相机转换到另一个坐标系了，和一般的nerf坐标系不同
    poses_centered = poses_centered @ BLENDER2OPENCV
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)
    print('center in center_poses',poses_centered[:, :3, 3].mean(0))

    return poses_centered, np.linalg.inv(pose_avg_homo) @ BLENDER2OPENCV
######################################

def parse_llff_pose(pose):
  """convert llff format pose to 4x4 matrix of intrinsics and extrinsics."""

  h, w, f = pose[:3, -1]
  c2w = pose[:3, :4]
  c2w_4x4 = np.eye(4)
  c2w_4x4[:3] = c2w
  c2w_4x4[:, 1:3] *= -1
  intrinsics = np.array(
      [[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  )
  return intrinsics, c2w_4x4


def batch_parse_llff_poses(poses):
  """Parse LLFF data format to opencv/colmap format."""
  all_intrinsics = []
  all_c2w_mats = []
  for pose in poses:
    intrinsics, c2w_mat = parse_llff_pose(pose)
    all_intrinsics.append(intrinsics)
    all_c2w_mats.append(c2w_mat)
  all_intrinsics = np.stack(all_intrinsics)
  all_c2w_mats = np.stack(all_c2w_mats)
  return all_intrinsics, all_c2w_mats

def _load_data_multicam(basedir, camera_indices, factor=None, load_imgs=True, frame_indices=[0, 74], downsample_scale=None, with_depth=False, cimle_dir=None, num_hypothesis=20):
  """Function for loading LLFF data."""
  poses_arr = np.load(os.path.join(basedir, 'poses_bounds_gl2llff.npy'))
  poses = poses_arr[:, :-2].reshape([-1, 3, 5])
  bds = poses_arr[:, -2:]
  
  # print()
  # print("In load data multicam.")
  # print(poses.shape)
  # print(bds.shape)
#   # print(bds[:,0])
#   # print(bds[:,10])
#   print(camera_indices)
#   print(len(camera_indices))
#   exit()

  imgdir = os.path.join(basedir, 'mv_images')

  if not os.path.exists(imgdir):
    print(imgdir, 'does not exist, returning')
    return
  
  imgfols = [
      os.path.join(imgdir, f)
      for f in sorted(os.listdir(imgdir))
  ]

  # print(imgfols)
  # print(len(imgfols))
  # exit()

  if frame_indices is not None:
    imgfols_selected = [imgfols[x] for x in frame_indices]
  else:
    imgfols_selected = imgfols

#   print(imgfols_selected)
#   print(len(imgfols_selected))
#   exit()

  '''
  Load for each camera
  '''
  all_poses = []
  all_bounds = []
  all_imgfiles = []

  for cam_idx in camera_indices:
    curr_imgfiles = [
      os.path.join(imgfols_selected[i], 'cam%02d.png'%cam_idx)
      for i in range(len(imgfols_selected))
    ]
    curr_poses = [
      poses[cam_idx]
      for _ in range(len(imgfols_selected))
    ]    
    curr_bounds = [
        bds[cam_idx]
      for _ in range(len(imgfols_selected))
    ]

    all_poses += curr_poses
    all_bounds += curr_bounds
    all_imgfiles += curr_imgfiles

  all_poses = np.stack(all_poses, 0)
  all_bounds = np.stack(all_bounds, 0)

  poses = all_poses.transpose([1, 2, 0])
  bds = all_bounds.transpose([1, 0])
  imgfiles = all_imgfiles

  # print(len(imgfiles))
  # print(imgfiles[0])
  # print(poses.shape)
  # print(bds.shape)
#   exit()

  if with_depth:
    ############################################    
    #### Load cimle depth maps ####
    ############################################    
    ## For now only for train poses
    leres_dir = os.path.join(basedir, "scade_hypothesis", cimle_dir)
    hypothesis_dir = sorted(os.listdir(leres_dir))

    hypothesis_dir_selected = [os.path.join(leres_dir, hypothesis_dir[x]) for x in frame_indices]

    all_depth_hypothesis = []

    for cam_idx in camera_indices:
      for i in range(len(hypothesis_dir_selected)):
        
        curr_depth_hypotheses = []
        curr_dir = hypothesis_dir_selected[i]

        for j in range(num_hypothesis):
          cimle_depth_name = os.path.join(curr_dir, 'cam%02d'%cam_idx +"_"+str(j)+".npy")        
          cimle_depth = np.load(cimle_depth_name).astype(np.float32)
          cimle_depth = cv2.resize(cimle_depth, (int(cimle_depth.shape[1]/downsample_scale), int(cimle_depth.shape[0]/downsample_scale)), interpolation=cv2.INTER_NEAREST)          

          # print(cimle_depth)
          # print(cimle_depth.shape)

          cimle_depth = np.expand_dims(cimle_depth, -1)
          curr_depth_hypotheses.append(cimle_depth)

        curr_depth_hypotheses = np.array(curr_depth_hypotheses)
        all_depth_hypothesis.append(curr_depth_hypotheses
        )
    all_depth_hypothesis = np.array(all_depth_hypothesis)
    print(all_depth_hypothesis.shape)
    # exit()

  if poses.shape[-1] != len(imgfiles):
    print(
        '{}: Mismatch between imgs {} and poses {} !!!!'.format(
            basedir, len(imgfiles), poses.shape[-1]
        )
    )
    raise NotImplementedError

  sh = imageio.imread(imgfiles[0]).shape
  poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
  poses[2, 4, :] = poses[2, 4, :]  # * 1. / factor

  def imread(f):
    convert_fn = cv2.COLOR_BGR2RGB
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, convert_fn)
    return img
    # if f.endswith('png'):
    #   return imageio.imread(f, ignoregamma=True)
    # else:
    #   return imageio.imread(f)

  if not load_imgs:
    imgs = None
  else:
    imgs = [(imread(f)[..., :3] / 255.0).astype(np.float32) for f in imgfiles]
    # imgs = np.stack(imgs, -1)
    # print('Loaded image data', imgs.shape, poses[:, -1, 0])

  all_images = []
  if downsample_scale is not None:
    for img in imgs:
      img = cv2.resize(img, (int(img.shape[1]/downsample_scale), int(img.shape[0]/downsample_scale)), interpolation=cv2.INTER_LINEAR)
      all_images.append(img)
  else:
    all_images = imgs

  all_images = np.stack(all_images, -1)
  print('Loaded image data', all_images.shape, poses[:, -1, 0])
  
  if not with_depth:
    return poses, bds, all_images, imgfiles
  else:
    return poses, bds, all_images, imgfiles, all_depth_hypothesis
   


def normalize(x):
  return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
  vec2 = normalize(z)
  vec1_avg = up
  vec0 = normalize(np.cross(vec1_avg, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, pos], 1)
  return m


def ptstocam(pts, c2w):
  tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
  return tt


def poses_avg(poses):
  hwf = poses[0, :3, -1:]

  center = poses[:, :3, 3].mean(0)
  vec2 = normalize(poses[:, :3, 2].sum(0))
  up = poses[:, :3, 1].sum(0)
  c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

  return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
  """Render a spiral path."""

  render_poses = []
  rads = np.array(list(rads) + [1.0])
  hwf = c2w[:, 4:5]

  for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
    c = np.dot(
        c2w[:3, :4],
        np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
        * rads,
    )
    z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
    render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
  return render_poses


def recenter_poses(poses):
  """Recenter camera poses into centroid."""
  poses_ = poses + 0
  bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
  c2w = poses_avg(poses)
  c2w = np.concatenate([c2w[:3, :4], bottom], -2)
  bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
  poses = np.concatenate([poses[:, :3, :4], bottom], -2)

  poses = np.linalg.inv(c2w) @ poses
  poses_[:, :3, :4] = poses[:, :3, :4]
  poses = poses_
  return poses


def load_llff_data_multicam(
    basedir,
    camera_indices,
    factor=8,
    render_idx=8,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    load_imgs=True,
    downsample=2.0,
    frame_indices = [0]
):
  """Load LLFF forward-facing data.
  
  Args:
    basedir: base directory
    camera_indices: select which cameras to use from the nvidia data
    factor: resize factor
    render_idx: rendering frame index from the video
    recenter: recentor camera poses
    bd_factor: scale factor for bounds
    spherify: spherify the camera poses
    load_imgs: load images from the disk

  Returns:
    images: video frames
    poses: corresponding camera parameters
    bds: bounds
    render_poses: rendering camera poses 
    i_test: test index
    imgfiles: list of image path
    scale: scene scale
  """
  all_camera_indices = np.arange(16)
  out = _load_data_multicam(
      basedir, all_camera_indices, factor=None, load_imgs=load_imgs, frame_indices=frame_indices, downsample_scale=downsample
  )

  if out is None:
    return
  else:
    poses, bds, imgs, imgfiles = out

  poses = np.moveaxis(poses, -1, 0)
  bds = np.moveaxis(bds, -1, 0)
  imgs = np.moveaxis(imgs, -1, 0)

  # print(poses.shape)
  # print(bds.shape)
  # print(imgfiles[10])
  print(imgs.shape)
  # exit()

  ##### Pose and bounds correction ###
  # Step 1: rescale focal length according to training resolution
  H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images

  ### Need to set this up if we want to resize
  focal = focal/downsample
  H = int(H/downsample)
  W = int(W/downsample)

  # Step 2: correct poses
  # Original poses has rotation in form "down right back", change to "right up back"
  # See https://github.com/bmild/nerf/issues/34
  poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
  # (N_images, 3, 4) exclude H, W, focal
  poses, pose_avg = center_poses(poses)
  # print('pose_avg in read_meta', self.pose_avg)
  # self.poses = poses @ self.blender2opencv

  # Step 3: correct scale so that the nearest depth is at a little more than 1.0
  # See https://github.com/bmild/nerf/issues/34
  near_original = bds.min()
  scale_factor = near_original * bd_factor  # 0.75 is the default parameter
  print('scale_factor', scale_factor)
  # the nearest depth is at 1/0.75=1.33
  bds /= scale_factor
  poses[..., 3] /= scale_factor

  near = np.min(bds[..., 0])*0.8
  far = np.max(bds[..., 1])*1.2  # focus on central object only

  # print(poses.shape)
  # print(bds.shape)
  # print(near)
  # print(far)
  # exit()
  ##############################

  all_imgs = []
  all_poses = []
  all_intrinsics = []
  
  for i in range(poses.shape[0]):

      c2w = torch.FloatTensor(poses[i])

      img = imgs[i]
      all_imgs.append(img)
      all_poses.append(poses[i])

      fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
      all_intrinsics.append(np.array((fx, fy, cx, cy)))


  all_imgs = np.array(imgs)
  all_poses = np.array(all_poses)
  all_intrinsics = np.array(all_intrinsics)

  if len(camera_indices) == 16:
    ### Fix this ####
    i_test = np.arange(3, poses.shape[0], 5)
    i_train = np.setdiff1d(np.arange(poses.shape[0]), i_test)
    # print(i_test)
    # print(i_train)
    
    i_split = [i_train, i_test]

    # print(i_split)
    # exit()
  else:
    i_train = np.array(camera_indices)
    i_test = np.setdiff1d(all_camera_indices, i_train)
    i_split = [i_train, i_test]

    # print(i_train)
    # print(i_test)


  spiral = True
  if spiral:
    print('================= render_path_spiral ==========================')
    c2w = poses_avg(poses)
    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * 0.9, bds.max() * 2.0
    dt = 0.75
    mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    focal = mean_dz * 1.5

    # Get radii for spiral path
    # shrink_factor = 0.8
    zdelta = close_depth * 0.2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 80, 0)
    c2w_path = c2w
    n_views = 120
    n_rots = 2

    print(c2w_path.shape)

    # Generate poses for spiral path
    render_poses = render_path_spiral(
        c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=n_rots, N=n_views
    )
  else:
    raise NotImplementedError

  render_poses = np.array(render_poses).astype(np.float32)

  print(render_poses.shape)


  
  print("=====Done loading data.=======")
  print(all_imgs.shape)
  print(all_poses.shape)
  print(all_intrinsics.shape)


  return all_imgs, None, None, all_poses, H, W, all_intrinsics, near, far, i_split, render_poses, None


def load_llff_data_multicam_withdepth(
    basedir,
    camera_indices,
    factor=8,
    render_idx=8,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    load_imgs=True,
    downsample=2.0,
    frame_indices = [0],
    cimle_dir = "dump",
    num_hypothesis = 20
):
  """Load LLFF forward-facing data.
  
  Args:
    basedir: base directory
    camera_indices: select which cameras to use from the nvidia data
    factor: resize factor
    render_idx: rendering frame index from the video
    recenter: recentor camera poses
    bd_factor: scale factor for bounds
    spherify: spherify the camera poses
    load_imgs: load images from the disk

  Returns:
    images: video frames
    poses: corresponding camera parameters
    bds: bounds
    render_poses: rendering camera poses 
    i_test: test index
    imgfiles: list of image path
    scale: scene scale
  """
  all_camera_indices = np.arange(16)
  out = _load_data_multicam(
      basedir, all_camera_indices, factor=None, load_imgs=load_imgs, frame_indices=frame_indices, downsample_scale=downsample, \
        with_depth=True, cimle_dir=cimle_dir, num_hypothesis = num_hypothesis
  )

  if out is None:
    return
  else:
    poses, bds, imgs, imgfiles, all_depth_hypothesis = out

  poses = np.moveaxis(poses, -1, 0)
  bds = np.moveaxis(bds, -1, 0)
  imgs = np.moveaxis(imgs, -1, 0)

  # print(poses.shape)
  # print(bds.shape)
  # print(imgfiles[10])
  print(imgs.shape)
  # exit()

  ##### Pose and bounds correction ###
  # Step 1: rescale focal length according to training resolution
  H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images

  ### Need to set this up if we want to resize
  focal = focal/downsample
  H = int(H/downsample)
  W = int(W/downsample)

  # Step 2: correct poses
  # Original poses has rotation in form "down right back", change to "right up back"
  # See https://github.com/bmild/nerf/issues/34
  poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
  # (N_images, 3, 4) exclude H, W, focal
  poses, pose_avg = center_poses(poses)
  # print('pose_avg in read_meta', self.pose_avg)
  # self.poses = poses @ self.blender2opencv

  # Step 3: correct scale so that the nearest depth is at a little more than 1.0
  # See https://github.com/bmild/nerf/issues/34
  near_original = bds.min()
  scale_factor = near_original * bd_factor  # 0.75 is the default parameter
  print('scale_factor', scale_factor)
  # the nearest depth is at 1/0.75=1.33
  bds /= scale_factor
  poses[..., 3] /= scale_factor

  near = np.min(bds[..., 0])*0.8
  far = np.max(bds[..., 1])*1.2  # focus on central object only

  # print(poses.shape)
  # print(bds.shape)
  # print(near)
  # print(far)
  # exit()
  ##############################

  all_imgs = []
  all_poses = []
  all_intrinsics = []
  
  for i in range(poses.shape[0]):

      c2w = torch.FloatTensor(poses[i])

      img = imgs[i]
      all_imgs.append(img)
      all_poses.append(poses[i])

      fx, fy, cx, cy = focal, focal, W/2.0, H/2.0
      all_intrinsics.append(np.array((fx, fy, cx, cy)))


  all_imgs = np.array(imgs)
  all_poses = np.array(all_poses)
  all_intrinsics = np.array(all_intrinsics)

  if len(camera_indices) == 16:
    ### Fix this ####
    i_test = np.arange(3, poses.shape[0], 5)
    i_train = np.setdiff1d(np.arange(poses.shape[0]), i_test)
    # print(i_test)
    # print(i_train)
    
    i_split = [i_train, i_test]

    # print(i_split)
    # exit()
  else:
    i_train = np.array(camera_indices)
    i_test = np.setdiff1d(all_camera_indices, i_train)
    i_split = [i_train, i_test]

    # print(i_train)
    # print(i_test)

  all_depth_hypothesis = np.clip(all_depth_hypothesis, near, far)

  ### Get video poses from the json file
  json_fname =  os.path.join(basedir, 'transforms_video.json')

  with open(json_fname, 'r') as fp:
      meta = json.load(fp)

  video_poses = []
  video_intrinsics = []

  for frame in meta['frames']:
    video_poses.append(np.array(frame['transform_matrix']))
    fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
    video_intrinsics.append(np.array((fx, fy, cx, cy)))

  # spiral = True
  # if spiral:
  #   print('================= render_path_spiral ==========================')
  #   c2w = poses_avg(poses)
  #   ## Get spiral
  #   # Get average pose
  #   up = normalize(poses[:, :3, 1].sum(0))

  #   # Find a reasonable "focus depth" for this dataset
  #   close_depth, inf_depth = bds.min() * 0.9, bds.max() * 2.0
  #   dt = 0.75
  #   mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
  #   focal = mean_dz * 1.5

  #   # Get radii for spiral path
  #   # shrink_factor = 0.8
  #   zdelta = close_depth * 0.2
  #   tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
  #   rads = np.percentile(np.abs(tt), 80, 0)
  #   c2w_path = c2w
  #   n_views = 120
  #   n_rots = 2

  #   print(c2w_path.shape)

  #   # Generate poses for spiral path
  #   render_poses = render_path_spiral(
  #       c2w_path, up, rads, focal, zdelta, zrate=0.5, rots=n_rots, N=n_views
  #   )
  # else:
  #   raise NotImplementedError

  video_poses = np.array(video_poses).astype(np.float32)
  video_intrinsics = np.array(video_intrinsics).astype(np.float32)

  # print(render_poses.shape)


  
  print("=====Done loading data.=======")
  print(all_imgs.shape)
  print(all_poses.shape)
  print(all_intrinsics.shape)


  return all_imgs, None, None, all_poses, H, W, all_intrinsics, near, far, i_split, video_poses, video_intrinsics, all_depth_hypothesis