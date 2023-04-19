```bash
# STEP 1
python download.py # download dataseA
```

# Generate Dataset with Random Samples at Different Distances
```bash
# Template
# blender -b lego.blend -y --python script_train_views.py -- <name> <number_of_views> <distance_ratio_near> <distance_ratio_far>

# Lego (Done)
blender -b lego.blend -y --python script_train_views.py -- lego 100 0.5 1.25 # done

```

# Generate Dataset with at Different Distances
To render multiple distances per scenes, use `script_train_views2.py`:
```bash
# Tempalte
# blender -b <scene>.blend -y --python script_train_views2.py -- <name> <n_views> <distance_ratio_near> <distance_ratio_far> <n_ratios_sampled>
# Lego
blender -b lego.blend -y --python script_train_views2.py -- lego_train 200 0.5 1.5 5
blender -b lego.blend -y --python script_train_views2.py -- lego_val 100 0.5 1.5 5 # done 
blender -b lego.blend -y --python script_train_views2.py -- lego_test 100 0.5 1.5 5 # done 

# Drums
blender -b drums.blend -y --python script_train_views2.py -- drums_train 200 0.5 1.5 5 # done 
blender -b drums.blend -y --python script_train_views2.py -- drums_val 100 0.5 1.5 5 # done
blender -b drums.blend -y --python script_train_views2.py -- drums_test 100 0.5 1.5 5 # done 

# Hotdog
blender -b hotdog.blend -y --python script_train_views2.py -- hotdog_train 200 0.5 1.5 5 # done 
blender -b hotdog.blend -y --python script_train_views2.py -- hotdog_val 100 0.5 1.5 5 # done 
blender -b hotdog.blend -y --python script_train_views2.py -- hotdog_test 100 0.5 1.5 5 # done 

# MIC
blender -b mic.blend -y --python script_train_views2.py -- mic_train 200 0.5 1.5 5 # done 
blender -b mic.blend -y --python script_train_views2.py -- mic_val 100 0.5 1.5 5 # done 
blender -b mic.blend -y --python script_train_views2.py -- mic_test 100 0.5 1.5 5 # done 

# TODO: ficus 
blender -b ficus.blend -y --python script_train_views2.py -- ficus_train 200 0.5 1.5 5 # running
blender -b ficus.blend -y --python script_train_views2.py -- ficus_val 100 0.5 1.5 5 # running
blender -b ficus.blend -y --python script_train_views2.py -- ficus_test 100 0.5 1.5 5 # running

# TODO: materials 
blender -b materials.blend -y --python script_train_views2.py -- materials_train 200 0.5 1.5 5 # running
blender -b materials.blend -y --python script_train_views2.py -- materials_val 100 0.5 1.5 5 # running
blender -b materials.blend -y --python script_train_views2.py -- materials_test 100 0.5 1.5 5 # running
```
