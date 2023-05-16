import os

import cv2
import numpy as np
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import shutil

def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return img, depth

def load_ground_truth_depth(basedir, train_filenames, image_size, depth_scaling_factor):
    H, W = image_size
    gt_depths = []
    gt_valid_depths = []
    for filename in train_filenames:
        filename = filename.replace("rgb", "target_depth")
        filename = filename.replace(".jpg", ".png")
        gt_depth_fname = os.path.join(basedir, filename)
        if os.path.exists(gt_depth_fname):
            gt_depth = cv2.imread(gt_depth_fname, cv2.IMREAD_UNCHANGED).astype(np.float64)
            gt_valid_depth = gt_depth > 0.5
            gt_depth = (gt_depth / depth_scaling_factor).astype(np.float32)
        else:
            gt_depth = np.zeros((H, W))
            gt_valid_depth = np.full_like(gt_depth, False)
        gt_depths.append(np.expand_dims(gt_depth, -1))
        gt_valid_depths.append(gt_valid_depth)
    gt_depths = np.stack(gt_depths, 0)
    gt_valid_depths = np.stack(gt_valid_depths, 0)
    return gt_depths, gt_valid_depths

def load_scene(basedir, original_imgs, ref_dir):
    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    filenames = []

    i_train = []
    idx_orig = []

    s = "train"

    json_fname =  os.path.join(ref_dir, 'transforms_{}.json'.format(s))

    with open(json_fname, 'r') as fp:
        meta = json.load(fp)

    ### Only take the file names from the current basedir
    json_fname =  os.path.join(basedir, 'transforms_{}.json'.format(s))

    with open(json_fname, 'r') as fp:
        meta_curr = json.load(fp)

    curr_fnames = []
    for curr_frame in meta_curr['frames']:
        curr_fnames.append(curr_frame['file_path'])

    near = float(meta['near'])
    far = float(meta['far'])
    depth_scaling_factor = float(meta['depth_scaling_factor'])
   
    imgs = []
    depths = []
    valid_depths = []
    poses = []
    intrinsics = []
    
    count = 0
    for i in range(len(meta['frames'])):
        frame = meta['frames'][i]

        if frame['file_path'] not in curr_fnames:
            print("Skipping " + frame['file_path'] + ".")
            continue

        if len(frame['file_path']) != 0 or len(frame['depth_file_path']) != 0:
            img, depth = read_files(basedir, frame['file_path'], frame['depth_file_path'])

            if depth.ndim == 2:
                depth = np.expand_dims(depth, -1)

            # valid_depth = depth[:, :, 0] > 0.5 # 0 values are invalid depth
            valid_depth = depth[:, :, 0] > 0.1 # 0 values are invalid depth
            depth = (depth / depth_scaling_factor).astype(np.float32)

            filenames.append(frame['file_path'])
            
            imgs.append(img)
            depths.append(depth)
            valid_depths.append(valid_depth)

        poses.append(np.array(frame['transform_matrix']))
        H, W = img.shape[:2]
        fx, fy, cx, cy = frame['fx'], frame['fy'], frame['cx'], frame['cy']
        intrinsics.append(np.array((fx, fy, cx, cy)))

        ### Check if it is the original image
        img_num = frame['file_path'].split("/")[-1][:-4]
        if img_num in original_imgs:
            idx_orig.append(count) 

        i_train.append(count)
        count += 1

    imgs = np.array(imgs)
    depths = np.array(depths)
    valid_depths = np.array(valid_depths)
    poses = np.array(np.array(poses).astype(np.float32))
    intrinsics = np.array(np.array(intrinsics).astype(np.float32))

    idx_orig = np.array(idx_orig)

    gt_depths, gt_valid_depths = load_ground_truth_depth(ref_dir, filenames, (H, W), depth_scaling_factor)
    
    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_train, gt_depths, gt_valid_depths, idx_orig

def get_K(intrinsics):
    # view direction corresponds to negative z direction
    K = torch.zeros((*intrinsics.shape[:-1], 4, 4))
    K[..., 0, 0] = intrinsics[..., 0]
    K[..., 1, 1] = intrinsics[..., 1]
    K[..., 0, 2] = -intrinsics[..., 2]
    K[..., 1, 2] = -intrinsics[..., 3]
    K[..., 2, 2] = -1.
    K[..., 3, 3] = 1.
    return K

def get_nearest(xy, image, valid_mask):
    H, W = image.shape
    r = (float(H) - xy[..., 1]).floor().clamp(0, H - 1).type(torch.LongTensor)
    c = xy[..., 0].floor().clamp(0, W - 1).type(torch.LongTensor)
    valid_result = valid_mask[r, c]
    r = r[valid_result]
    c = c[valid_result]
    result = image[r, c]
    return result, valid_result

def get_pixel_size_in_world(xy, depth, img2cam):
    xyz_low = torch.cat((xy - 0.5, torch.ones_like(depth.unsqueeze(-1))), -1) * depth.unsqueeze(-1)
    xyz_high = torch.cat((xy + 0.5, torch.ones_like(depth.unsqueeze(-1))), -1) * depth.unsqueeze(-1)
    xyz_low_cam = torch.matmul(img2cam[:3, :3], xyz_low.unsqueeze(-1)).squeeze(-1)
    xyz_high_cam = torch.matmul(img2cam[:3, :3], xyz_high.unsqueeze(-1)).squeeze(-1)
    dxy = xyz_high_cam[..., :2] - xyz_low_cam[..., :2]
    return dxy.pow(2).sum(-1).sqrt()

# Assumption on depth error in meters like eq. 6 in https://openaccess.thecvf.com/content_cvpr_2013/papers/Barron_Intrinsic_Scene_Properties_2013_CVPR_paper.pdf
def expected_depth_error(z):
    return 1.5e-3 * z.pow(2) + 0.03

def count_occurrence(i_0, i_others, depths, valid_depths, rowcol2xyz, img2cam, world2img, img2world, device, distance_limit=None):
    H, W = depths.shape[1:3]
    occurrence_count = torch.zeros_like(depths[i_0], dtype=int)

    valid_depth_0 = valid_depths[i_0]
    depth_0 = depths[i_0]

    # backproject points with valid depth from image 0 to the world
    xyz_0 = rowcol2xyz[valid_depth_0] * depth_0[valid_depth_0]
    if distance_limit is not None:
        is_in_distance_limit_0 = xyz_0[:, 2] <= distance_limit
    else:
        is_in_distance_limit_0 = torch.ones_like(xyz_0[:, 2]).bool()
    
    xyz_0 = xyz_0[is_in_distance_limit_0]
    xyzh_0 = torch.cat((xyz_0, torch.ones((*xyz_0.shape[:-1], 1)).to(device)), -1)

    max_overlap = 0
    # print(i_0)
    # print(i_others)
    # print(depths.shape)

    for i_1 in i_others:
        valid_depth_1 = valid_depths[i_1]
        depth_1 = depths[i_1]

        # project points in the world to image 1
        xyz_world = torch.matmul(img2world[i_0], xyzh_0.unsqueeze(-1))
        xyzh_1 = torch.matmul(world2img[i_1], xyz_world).squeeze()

        # check if points are in front of the camera
        z_1 = xyzh_1[..., 2]
        is_infront_of_cam_1 = z_1 > 0.
        z_1 = z_1[is_infront_of_cam_1]
        xyzh_1 = xyzh_1[is_infront_of_cam_1]

        # check if points are within height and width of image 1
        xy_1 = (xyzh_1[..., :2] / z_1.unsqueeze(-1))
        is_in_image_bounds = torch.logical_and(xy_1[..., 0] < float(W), xy_1[..., 0] >= 0.) # x bound
        is_in_image_bounds = torch.logical_and(torch.logical_and(xy_1[..., 1] <= float(H), xy_1[..., 1] > 0.), is_in_image_bounds) # y bound
        xy_1 = xy_1[is_in_image_bounds]
        if torch.numel(xy_1) == 0:
            continue            

        # get depth in image 1 and check if it is valid
        z_measured_1, has_valid_depth_1 = get_nearest(xy_1, depth_1.squeeze(-1), valid_depth_1)
        if distance_limit is not None:
            is_in_distance_limit_1 = z_measured_1 <= distance_limit
        else:
            is_in_distance_limit_1 = torch.ones_like(z_measured_1).bool()
        z_measured_1 = z_measured_1[is_in_distance_limit_1]
        z_measured_0 = depth_0[valid_depth_0][is_in_distance_limit_0][is_infront_of_cam_1][is_in_image_bounds][has_valid_depth_1][is_in_distance_limit_1].squeeze(-1)
        if torch.numel(z_measured_0) == 0:
            continue

        # assume an error depending of the real world size corresponding to the pixels
        threshold_1 = get_pixel_size_in_world(xy_1[has_valid_depth_1][is_in_distance_limit_1], z_measured_1, img2cam[i_1])
        threshold_0 = get_pixel_size_in_world(rowcol2xyz[..., :2][valid_depth_0][is_in_distance_limit_0][is_infront_of_cam_1][is_in_image_bounds][has_valid_depth_1][is_in_distance_limit_1], \
            z_measured_0, img2cam[i_0])
        threshold = threshold_0 + threshold_1
        # add error depending on the expected measurement error of both measurements
        threshold = threshold + expected_depth_error(z_measured_0) + expected_depth_error(z_measured_1)
        #threshold = 0.1 * torch.maximum(z_measured_0, z_measured_1)

        # check if 2d point pairs describe the same 3d point
        z_predicted = z_1[is_in_image_bounds][has_valid_depth_1][is_in_distance_limit_1]
        has_similar_3d_point = (z_predicted - z_measured_1).abs() < threshold

        # update occurrence count
        mask = torch.nonzero(valid_depth_0.view(-1))[is_in_distance_limit_0][is_infront_of_cam_1][is_in_image_bounds][has_valid_depth_1][is_in_distance_limit_1][has_similar_3d_point]
        occurrence_count.view(-1)[mask] += 1
        max_overlap = max(max_overlap, len(mask) / float(valid_depth_0.sum()))

    # print(occurrence_count.sum())
    # print()

    return occurrence_count, max_overlap

def main():
    ref_dir = "/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/7/"
    # ref_dir = "/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/7/"
    # ref_dir = "/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/7/"

    # ## Original scene
    # dataset_dir = "scene0781_00"
    # data_dir = "/orion/group/scannet_v2/dense_depth_priors/scenes/"

    # dataset_dir = "scene0758_00"
    # data_dir = "/orion/group/scannet_v2/dense_depth_priors/scenes/"
    
    # dataset_dir = "scene0710_00"
    # data_dir = "/orion/group/scannet_v2/dense_depth_priors/scenes/"


    dataset_dir = "24_1"

    ### For scene710
    # data_dir = "/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene710/"
    # original_imgs = ["1000", "1074", "1118", "1185", "1255", "1319", "1415", "1461", "1568", "1670", "1712", "1759", "1796", "458", "495", "663", "872", "932"]

    ### For scene758
    # data_dir = "/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene758/"
    # original_imgs = ["105", "1349", "1423", "1449", "1498", "1542", "1669", "1700", "1719", "1769", "1819", "467", "519", "590", "703", "720", "759", "850", "869", "935"]

    # ### For scene781
    # data_dir = "/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense/scene781/"
    data_dir = "/orion/group/scannet_v2/dense_depth_priors/rebuttal_sparsity_v2/processed/sparsity_subsets_dense_shuffle/scene781/"
    original_imgs = ["0", "1030", "1070", "1253", "1313", "1689", "1740", "1851", "1941", "1972", "198", "2059", "420", "494", "620", "706", "784", "892"]
    

    dataset_type = "train"
    show_legend = True

    input_dir = os.path.join(data_dir, dataset_dir)
    output_dir = os.path.join(input_dir, 'occurrence_count')
    
    _, depths, valid_depths, poses, H, W, intrinsics, _, _, i_train, gt_depths, gt_valid_depths, idx_orig = load_scene(input_dir, original_imgs, ref_dir)


    # use ground truth if available
    if gt_depths is not None:
        print("Using ground truth depth")
        depths = gt_depths
        valid_depths = gt_valid_depths

    i_imgs = i_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths = torch.Tensor(depths[i_train]).to(device)
    valid_depths = torch.Tensor(valid_depths[i_train]).bool().to(device)
    poses = torch.Tensor(poses[i_train]).to(device)

    intrinsics = torch.Tensor(intrinsics[i_train]).to(device)

    world2cam = poses.inverse()
    cam2img = get_K(intrinsics).to(device)
    img2cam = cam2img.inverse()
    world2img = torch.matmul(cam2img, world2cam)
    img2world = world2img.inverse()
    col, row = torch.meshgrid(torch.tensor(range(W)), torch.tensor(range(H)))
    rowcol2xy = torch.stack((col.t() + 0.5, H - (row.t() + 0.5)), -1)
    rowcol2xyz = torch.cat((rowcol2xy, torch.ones(H, W, 1)), -1).to(device)
    
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    sum_valid_pixels = 0
    sum_occurences = 0

    sum_pixels_seen_once = 0
    sum_pixels_seen_at_least_twice = 0
    sum_pixels_seen_at_least_thrice = 0


    max_overlaps = []

    print(len(idx_orig))
    print(len(i_train))

    for i in idx_orig:
        ### To get the original set
        # occurrence_img, max_overlap = count_occurrence(i, idx_orig, depths, valid_depths, rowcol2xyz, img2cam, world2img, img2world, device)

        occurrence_img, max_overlap = count_occurrence(i, i_train, depths, valid_depths, rowcol2xyz, img2cam, world2img, img2world, device)

        max_overlaps.append(max_overlap)

        sum_occurences += occurrence_img[valid_depths[i]].sum()
        sum_valid_pixels += valid_depths[i].sum()

        sum_pixels_seen_once += (occurrence_img[valid_depths[i]] < 2).sum()
        sum_pixels_seen_at_least_twice += (occurrence_img[valid_depths[i]] >= 2).sum()
        sum_pixels_seen_at_least_thrice += (occurrence_img[valid_depths[i]] >= 3).sum()


        # write image
        occurrence_img[occurrence_img > 3] = 4
        occurrence_img = occurrence_img.cpu().numpy().astype(np.uint8)
        occurrence_img = (occurrence_img.astype(float) / 4. * 255.).astype(np.uint8)
        occurrence_img = cv2.applyColorMap(occurrence_img, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(os.path.join(output_dir, "{}.jpg".format(i)), occurrence_img)

    print("Pixels in {} images are observed in {} train images on average.".format(dataset_type, sum_occurences / sum_valid_pixels))
    print("{} of the pixels are seen once in the train images.".format(sum_pixels_seen_once / sum_valid_pixels))
    print("{} of the pixels are seen at least twice in the train images.".format(sum_pixels_seen_at_least_twice / sum_valid_pixels))
    print("{} of the pixels are seen at least thrice in the train images.".format(sum_pixels_seen_at_least_thrice / sum_valid_pixels))


    # print("{} images have on average {} overlap to the most overlapping train image".format(dataset_type, np.mean(np.array(max_overlaps))))

    # if show_legend:
    #     legend_entries = ['0 times', '1 time', '2 times', '3 times', '>= 4 times']
    #     cols = [cm.viridis(0), cm.viridis(0.25), cm.viridis(0.5), cm.viridis(0.75), cm.viridis(1.)]
    #     legend_handles = []
    #     for col, lab in zip(cols, legend_entries):
    #         legend_handles.append(mpatches.Patch(color=col, label=lab))
    #     plt.figure()
    #     plt.legend(handles=legend_handles, prop={"size" : 20})
    #     plt.show()

if __name__=='__main__':
    main()