```bash
# STEP 1
python download.py # download dataseA
```

```bash
# Template
# blender -b mic.blend -y --python script_train_views.py -- <name> <number_of_views> <distance_ratio_near> <distance_ratio_far>
blender -b mic.blend -y --python script_train_views.py -- mic 100 0.75 1.25
```

To render multiple distances per scenes, use `script_train_views2.py`:
```bash
# Tempalte
# blender -b <scene>.blend -y --python script_train_views2.py -- <name> <n_views> <distance_ratio_near> <distance_ratio_far> <n_ratios_sampled>
blender -b lego.blend -y --python script_train_views2.py -- lego_train 200 0.5 1.5 5
```
