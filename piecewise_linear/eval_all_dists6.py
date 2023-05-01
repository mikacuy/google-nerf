import os, sys

all_test_dist = [0.5, 0.75, 1.0, 1.25]

# model_params = {0: {"scene_id": "lego_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"lego_linear_c128_i64_midpoint",
# 					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
# 				1: {"scene_id": "chair_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"chair_linear_c128_i64_midpoint",
# 					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
# 				2: {"scene_id": "hotdog_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"hotdog_linear_c128_i64_midpoint",
# 					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
# 				3: {"scene_id": "lego_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"lego_constant_c128_i64 ",
# 					"mode":"constant", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
# 				4: {"scene_id": "chair_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"chair_constant_c128_i64 ",
# 					"mode":"constant", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
# 				5: {"scene_id": "hotdog_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"hotdog_constant_c128_i64 ",
# 					"mode":"constant", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4}																											
# 				}

# model_params = {2: {"scene_id": "hotdog_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"hotdog_linear_c128_i64_midpoint",
# 					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
# 				3: {"scene_id": "lego_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"lego_constant_c128_i64 ",
# 					"mode":"constant", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
# 				4: {"scene_id": "chair_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"chair_constant_c128_i64 ",
# 					"mode":"constant", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
# 				5: {"scene_id": "hotdog_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
# 					"ckpt_dir":"log_multidist_perpose", "expname":"hotdog_constant_c128_i64 ",
# 					"mode":"constant", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4}																											
# 				}

model_params = {6: {"scene_id": "mic_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_multidist_perpose", "expname":"mic_linear_c128_i64_midpoint",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				7: {"scene_id": "mic_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_multidist_perpose", "expname":"mic_constant_c128_i64 ",
					"mode":"constant", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				8: {"scene_id": "ficus_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_multidist_perpose", "expname":"ficus_linear_c128_i64_midpoint",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				9: {"scene_id": "ficus_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_multidist_perpose", "expname":"ficus_constant_c128_i64 ",
					"mode":"constant", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},																											
				}

for model_idx in model_params:
	for test_dist in all_test_dist:

		command = "python3 eval_nerf_fixed_dist_mult.py test --scene_id {} \
		--data_dir {} --dataset blender \
		--depth_prior_network_path depth_priors_network_weights/20211027_092436.tar \
		--ckpt_dir {} --expname={} \
		--depth_loss_weight=-1.0 --N_samples={} --N_importance={} --input_ch_cam=0 \
		--mode={} --color_mode={} --set_near_plane={} --test_dist={}".format(model_params[model_idx]["scene_id"], \
			model_params[model_idx]["data_dir"], model_params[model_idx]["ckpt_dir"], model_params[model_idx]["expname"], \
			str(model_params[model_idx]["N_samples"]), str(model_params[model_idx]["N_importance"]), model_params[model_idx]["mode"],\
			model_params[model_idx]["color_mode"], str(model_params[model_idx]["set_near_plane"]), str(test_dist))

		os.system(command)

	print("Done with " + model_params[model_idx]["scene_id"] + ".")

