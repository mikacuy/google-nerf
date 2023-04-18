```bash
# STEP 1
python download.py # download dataseA
```

# Generate Dataset with Random Samples at Different Distances
```bash
# Template
# blender -b mic.blend -y --python script_train_views.py -- <name> <number_of_views> <distance_ratio_near> <distance_ratio_far>
blender -b mic.blend -y --python script_train_views.py -- mic 100 0.75 1.25
```

# Generate Dataset with at Different Distances
To render multiple distances per scenes, use `script_train_views2.py`:
```bash
# Tempalte
# blender -b <scene>.blend -y --python script_train_views2.py -- <name> <n_views> <distance_ratio_near> <distance_ratio_far> <n_ratios_sampled>
blender -b lego.blend -y --python script_train_views2.py -- lego_train 200 0.5 1.5 5
blender -b lego.blend -y --python script_train_views2.py -- lego_val 100 0.5 1.5 5 # rerun
blender -b lego.blend -y --python script_train_views2.py -- lego_test 100 0.5 1.5 5 # rerun

# TODO: need to render
blender -b drums.blend -y --python script_train_views2.py -- drums_train 200 0.5 1.5 5 # running
blender -b drums.blend -y --python script_train_views2.py -- drums_val 100 0.5 1.5 5 # running
blender -b drums.blend -y --python script_train_views2.py -- drums_test 100 0.5 1.5 5 # running

# TODO: still haven't render
blender -b hotdog.blend -y --python script_train_views2.py -- hotdog_train 200 0.5 1.5 5 # running
blender -b hotdog.blend -y --python script_train_views2.py -- hotdog_val 100 0.5 1.5 5 # running
blender -b hotdog.blend -y --python script_train_views2.py -- hotdog_test 100 0.5 1.5 5 # running

# TODO: still need to render
blender -b mic.blend -y --python script_train_views2.py -- mic_train 200 0.5 1.5 5 # running
blender -b mic.blend -y --python script_train_views2.py -- mic_val 100 0.5 1.5 5 # running
blender -b mic.blend -y --python script_train_views2.py -- mic_test 100 0.5 1.5 5 # running
```
