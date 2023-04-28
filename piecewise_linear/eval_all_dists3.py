import os, sys

all_test_dist = [0.5, 0.75, 1.0, 1.25]

model_params = {8: {"scene_id": "lego_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_harder", "expname":"lego_0.25_1.5_linear_c128_i64",
					"mode":"linear", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				9: {"scene_id": "chair_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_harder", "expname":"chair_0.25_1.5_linear_c128_i64",
					"mode":"linear", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				10: {"scene_id": "ship_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_harder", "expname":"ship_0.25_1.5_linear_c128_i64",
					"mode":"linear", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				11: {"scene_id": "hotdog_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_harder", "expname":"hotdog_0.25_1.5_linear_c128_i64",
					"mode":"linear", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				12: {"scene_id": "drums_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_harder", "expname":"drums_0.25_1.5_linear_c128_i64",
					"mode":"linear", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				13: {"scene_id": "materials_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_harder", "expname":"materials_0.25_1.5_linear_c128_i64",
					"mode":"linear", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				14: {"scene_id": "mic_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_harder", "expname":"mic_0.25_1.5_linear_c128_i64_lowerlr",
					"mode":"linear", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4},
				15: {"scene_id": "ficus_fixdist_nv100_dist0.5-1.5-5", "data_dir": "nerf_synthetic/fixed_dist_new/",\
					"ckpt_dir":"log_new_blender_closeup_harder", "expname":"ficus_0.25_1.5_linear_c128_i64_lowerlr",
					"mode":"linear", "color_mode":"left", "N_samples": 128, "N_importance": 64, "set_near_plane": 1e-4}																																																						
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

		os.system(command)

	print("Done with " + model_params[model_idx]["scene_id"] + ".")


	