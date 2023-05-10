```bash
# STEP 1
python download.py # download dataseA
```

# Generate Dataset with Random Samples at Different Distances
```bash
# Template
# blender -b lego.blend -y --python script_train_views.py -- <name> <number_of_views> <distance_ratio_near> <distance_ratio_far>

# Lego (Done)
blender -b lego.blend -y --python script_randview.py -- lego 100 0.5 1.25 # done
```

# Generate Dataset with Multipl Views Per Camera Poses
To render multiple distances per pose per scene, use `script_randdist.py`:
```bash
# Tempalte
# blender -b <scene>.blend -y --python script_multidist.py -- <name> <n_poses> <distance_ratio_near> <distance_ratio_far> <n_ratios_sampled_per_pose> <--randist?>
blender -b lego.blend -y --python script_multidist.py -- lego 50 0.5 1.5 5 --randdist
```

# Automation scripts
```bash
for f in chair drums ficus hotdog lego materials mic; do 
    echo $name; 
    blender -b $name.blend -y --python script_randview.py -- $name_test 1 0.5 1.0 --render_depth --render_sfn& 
done
```
