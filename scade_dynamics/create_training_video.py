import os, sys
import numpy
import imageio
from natsort import natsorted 

import PIL.Image
from PIL import Image

# video_frames_dir = "log_0918_chair_and_hotdog_motion/chair_and_hotdog_e1e0_mu10.0_std0.001_sxyz0.01_sdino1.0_fw25.0_ispcaTrue_pcadim3_mlrate1e-06/training_visu"
# outname = "chair_and_hotdog_training_evolution_potential2_lre6_std001.mp4"

# video_frames_dir = "log_0919_hotdog_motion_edited1/hotdogsplatepre_0edited1_mu100.0_std0.001_sxyz0.01_sdino1.0_fw25.0_ispcaTrue_pcadim3_mlrate1e-06/training_visu"
# outname = "pre_hotdog_training_evolution_lre6_std001.mp4"

video_frames_dir = "log_1017_l2_cache_tune/hotdogsplatepre_0edited1_recache100_mu100.0_std0.001_sxyz0.01_sdino1.0_fw25.0_ispcaTrue_pcadim3_mlrate5e-05_rcache10/training_visu"
outname = "pre_hotdog_training_evolution_caching.mp4"

res = []
# Iterate directory
for file in os.listdir(video_frames_dir):
    if file.endswith('_database_pcflowed_1to2.png'):
    # if file.endswith('_potential1.png'):
    # if file.endswith('_potential2.png'):
    # if file.endswith('_nn.png'):
        res.append(file)
res = natsorted(res)
print(len(res))

imageio.mimsave(outname,
                [imageio.imread(os.path.join(video_frames_dir, img)) for img in res],
                fps=100, macro_block_size=1)