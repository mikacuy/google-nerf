python run_nerf_nodepth.py train --ckpt_dir log_test --expname walking_test --camera_indices 0,2,3,10,11 --frame_idx=10


python run_motion_optimization_0902.py train --dataset blender --ckpt_dir log_0903_motion_pair --expname hotdog_pair --data_dir /home/mikacuy/coord-mvs/hotdog_data_v1/ --scene_id hotdog_single_shadowfix --use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 --feature_dir hotdog_single_shadowfix_dino_features_small --downsample=8 --i_print=1

python run_motion_optimization_0902.py train --dataset blender --ckpt_dir log_0903_motion_pair_downsamplepretrained --expname hotdog_pair --data_dir /home/mikacuy/coord-mvs/hotdog_data_v1/ --scene_id hotdog_single_shadowfix --use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 --feature_dir hotdog_single_shadowfix_dino_features_small --downsample=8 --pretrained_dir /home/mikacuy/coord-mvs/google-nerf/scade_dynamics/log_blender_withdepth_dino_downsample8/

python run_motion_optimization_0902.py train --dataset blender --ckpt_dir log_0903_motion_pair_skipview10 --expname hotdog_pair --data_dir /home/mikacuy/coord-mvs/hotdog_data_v1/ --scene_id hotdog_single_shadowfix --use_depth=1 --space_carving_weight 0.007 --freeze_ss 0 --feat_dim 384 --feature_dir hotdog_single_shadowfix_dino_features_small --downsample=8 --skip_views 10
