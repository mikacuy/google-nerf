#!/bin/bash

python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0918_hotdog_motion_overnightruns  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1 hotdog_single_plate_detached --scene_id2  hotdog_single_plate_detached_edited4 \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 \
--feature_dir1 hotdog_single_plate_detached_dino_features_small --feature_dir2 hotdog_single_plate_detached_edited4_dino_features_small \
--pretrained_fol1 hotdog_single_plate_detached --pretrained_fol2 hotdog_single_plate_detached_edited4 \
--downsample=4 --pnm_std 0.001 --pnm_mean 100.0 --expname hotdogsplatepre_0edited4 --feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-5 --visu 1 --num_iterations 100000 --num_y_to_sample 1024

python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0918_hotdog_motion_overnightruns  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1 hotdog_single_plate_detached --scene_id2  hotdog_single_plate_detached_edited4 \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 \
--feature_dir1 hotdog_single_plate_detached_dino_features_small --feature_dir2 hotdog_single_plate_detached_edited4_dino_features_small \
--pretrained_fol1 hotdog_single_plate_detached --pretrained_fol2 hotdog_single_plate_detached_edited4 \
--downsample=4 --pnm_std 0.001 --pnm_mean 100.0 --expname hotdogsplatepre_0edited4 --feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-6 --visu 1 --num_iterations 100000 --num_y_to_sample 1024

python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0918_hotdog_motion_overnightruns  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1 hotdog_single_plate_detached --scene_id2  hotdog_single_plate_detached_edited4 \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 \
--feature_dir1 hotdog_single_plate_detached_dino_features_small --feature_dir2 hotdog_single_plate_detached_edited4_dino_features_small \
--pretrained_fol1 hotdog_single_plate_detached --pretrained_fol2 hotdog_single_plate_detached_edited4 \
--downsample=4 --pnm_std 0.001 --pnm_mean 100.0 --expname hotdogsplatepre_0edited4 --feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-7 --visu 1 --num_iterations 150000 --num_y_to_sample 1024


python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0918_chair_and_hotdog_motion  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1  chair_and_hotdog_edited --scene_id2   chair_and_hotdog \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 -\
-feature_dir1 chair_and_hotdog_edited_dino_features_small --feature_dir2 chair_and_hotdog_dino_features_small \
--pretrained_fol1 chair_and_hotdog_edited --pretrained_fol2 chair_and_hotdog \
--downsample=4 --pnm_std 0.001 --pnm_mean 10.0 --expname chair_and_hotdog_e1e0 \--feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-5 --visu 1 --num_iterations 100000 --num_y_to_sample 1024

python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0918_chair_and_hotdog_motion  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1  chair_and_hotdog_edited --scene_id2   chair_and_hotdog \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 -\
-feature_dir1 chair_and_hotdog_edited_dino_features_small --feature_dir2 chair_and_hotdog_dino_features_small \
--pretrained_fol1 chair_and_hotdog_edited --pretrained_fol2 chair_and_hotdog \
--downsample=4 --pnm_std 0.001 --pnm_mean 10.0 --expname chair_and_hotdog_e1e0 \--feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-6 --visu 1 --num_iterations 100000 --num_y_to_sample 1024

python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0918_chair_and_hotdog_motion  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1  chair_and_hotdog_edited --scene_id2   chair_and_hotdog \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 -\
-feature_dir1 chair_and_hotdog_edited_dino_features_small --feature_dir2 chair_and_hotdog_dino_features_small \
--pretrained_fol1 chair_and_hotdog_edited --pretrained_fol2 chair_and_hotdog \
--downsample=4 --pnm_std 0.001 --pnm_mean 10.0 --expname chair_and_hotdog_e1e0 \--feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-7 --visu 1 --num_iterations 150000 --num_y_to_sample 1024


######################
python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0919_hotdog_motion_edited1  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1 hotdog_single_plate_detached --scene_id2  hotdog_single_plate_detached_edited1 \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 \
--feature_dir1 hotdog_single_plate_detached_dino_features_small --feature_dir2 hotdog_single_plate_detached_edited1_dino_features_small \
--pretrained_fol1 hotdog_single_plate_detached --pretrained_fol2 hotdog_single_plate_detached_edited1 \
--downsample=4 --pnm_std 0.001 --pnm_mean 100.0 --expname hotdogsplatepre_0edited1 --feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-6 --visu 1 --num_iterations 100000 --num_y_to_sample 1024


python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0918_chair_and_hotdog_motion  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1  chair_and_hotdog_edited --scene_id2   chair_and_hotdog \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 -\
-feature_dir1 chair_and_hotdog_edited_dino_features_small --feature_dir2 chair_and_hotdog_dino_features_small \
--pretrained_fol1 chair_and_hotdog_edited --pretrained_fol2 chair_and_hotdog \
--downsample=4 --pnm_std 0.001 --pnm_mean 10.0 --expname chair_and_hotdog_e1e0 \--feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-6 --visu 1 --num_iterations 100000 --num_y_to_sample 1024

python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0919_chair_and_hotdog_motion_flipped  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1  chair_and_hotdog --scene_id2   chair_and_hotdog_edited \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 -\
-feature_dir1 chair_and_hotdog_dino_features_small --feature_dir2 chair_and_hotdog_edited_dino_features_small \
--pretrained_fol1 chair_and_hotdog --pretrained_fol2 chair_and_hotdog_edited \
--downsample=4 --pnm_std 0.001 --pnm_mean 10.0 --expname chair_and_hotdog_e0e1 \--feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-6 --visu 1 --num_iterations 100000 --num_y_to_sample 1024




python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0920_chair_and_hotdog_motion  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1  chair_and_hotdog_edited --scene_id2   chair_and_hotdog \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 -\
-feature_dir1 chair_and_hotdog_edited_dino_features_small --feature_dir2 chair_and_hotdog_dino_features_small \
--pretrained_fol1 chair_and_hotdog_edited --pretrained_fol2 chair_and_hotdog \
--downsample=4 --pnm_std 0.001 --pnm_mean 10.0 --expname chair_and_hotdog_e1e0 \--feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-6 --visu 1 --num_iterations 100000 --num_y_to_sample 1024

python run_motion_optimization_gaussiannoise_twoway_pcaenabled_precompute_0917.py train --dataset blender \
--ckpt_dir log_0920_hotdog_motion_edited1  --data_dir /home/mikacuy/coord-mvs/demo_scenes_v1/  \
--scene_id1 hotdog_single_plate_detached --scene_id2  hotdog_single_plate_detached_edited1 \
--use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 \
--feature_dir1 hotdog_single_plate_detached_dino_features_small --feature_dir2 hotdog_single_plate_detached_edited1_dino_features_small \
--pretrained_fol1 hotdog_single_plate_detached --pretrained_fol2 hotdog_single_plate_detached_edited1 \
--downsample=4 --pnm_std 0.001 --pnm_mean 100.0 --expname hotdogsplatepre_0edited1 --feat_dist_weight 25.0 --is_dino_pca 1 \
--pcadim 3 --xyz_potential_scale 0.01 --motion_lrate 1e-6 --visu 1 --num_iterations 100000 --num_y_to_sample 1024