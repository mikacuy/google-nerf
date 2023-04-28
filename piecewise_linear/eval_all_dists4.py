import os, sys

all_test_dist = [0.5, 0.75, 1.0, 1.25]

model_params = {16: {"scene_id": "lego_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_midpointcolor", "expname":"lego_0.5_1.25_linear_c128_i64",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 0.5},				
				17: {"scene_id": "ship_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_midpointcolor", "expname":"ship_0.5_1.25_linear_c128_i64",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 0.5},
				18: {"scene_id": "hotdog_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_midpointcolor", "expname":"hotdog_0.5_1.25_linear_c128_i64",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 0.5},
				19: {"scene_id": "chair_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_midpointcolor", "expname":"chair_0.5_1.25_linear_c128_i64",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 0.5},
				20: {"scene_id": "drums_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_midpointcolor", "expname":"drums_0.5_1.25_linear_c128_i64",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 0.5},
				21: {"scene_id": "materials_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_midpointcolor", "expname":"materials_0.5_1.25_linear_c128_i64",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 0.5},
				22: {"scene_id": "mic_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_midpointcolor", "expname":"mic_0.5_1.25_linear_c128_i64_lowerlr",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 0.5},
				23: {"scene_id": "ficus_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_midpointcolor", "expname":"ficus_0.5_1.25_linear_c128_i64_lowerlr",
					"mode":"linear", "color_mode":"midpoint", "N_samples": 128, "N_importance": 64, "set_near_plane": 0.5},																														
				}

for model_idx in model_params:
	for test_dist in all_test_dist:

		command = "python3 eval_nerf_fixed_dist.py test --scene_id {} \
		--data_dir {} --dataset blender \
		--depth_prior_network_path depth_priors_network_weights/20211027_092436.tar \
		--ckpt_dir {} --expname={} \
		--depth_loss_weight=-1.0 --N_samples={} --N_importance={} --input_ch_cam=0 \
		--mode={} --color_mode={} --set_near_plane={} --test_dist={}".format(model_params[model_idx]["scene_id"], \
			model_params[model_idx]["data_dir"], model_params[model_idx]["ckpt_dir"], model_params[model_idx]["expname"], \
			str(model_params[model_idx]["N_samples"]), str(model_params[model_idx]["N_importance"]), model_params[model_idx]["mode"],\
			model_params[model_idx]["color_mode"], str(model_params[model_idx]["set_near_plane"]), str(test_dist))

		print(command)
		exit()

		os.system(command)

	print("Done with " + model_params[model_idx]["scene_id"] + ".")


	