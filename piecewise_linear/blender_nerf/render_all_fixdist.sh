#! /bin/bash

nviews=${1}
near=${2}
far=${3}
nsamples=${4}
# for name in chair drums ficus hotdog lego materials mic ship; do
# for name in chair drums ficus hotdog lego materials mic; do
# for name in ship; do
# for name in chair drums ficus hotdog lego materials mic ship; do
for name in lego; do
    echo $name;
    (
    blender -b $name.blend -y --python script_fix_dist.py \
        -- ${name}_rgba ${nviews} ${near} ${far} ${nsamples} \
        --test_only \
        --render_depth \
        --render_sfn;
    out_name=nerf_dataset/${name}_rgba_fixdist_nv${nviews}_dist${near}-${far}-${nsamples}_depth_sfn;
    for f in `ls -d ${out_name}/*/`; do
        python post_process_rgb.py $f --out_dir $f --keep_alpha_channel;
        # nerf_dataset/${name}_randdist_nv${nviews}_dist${near}-${far}
    done;
    zip -r ${out_name}.zip $out_name/
    ) &
done
