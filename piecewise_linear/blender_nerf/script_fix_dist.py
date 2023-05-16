# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import bpy
import argparse, sys, os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='NeRF training set.')
parser.add_argument('name', type=str, help='Name')
parser.add_argument('views', type=int, help='Number of views')
parser.add_argument('near', type=float, help='Distance ratio (near)')
parser.add_argument('far', type=float, help='Distance ratio (far)')
parser.add_argument('n_smp', type=int, default=5, help='Number of different distance samples')
parser.add_argument('--seed', type=int, default=None, help='RAndom seed.')
parser.add_argument('--test_only', action="store_true", help='RAndom seed.')
parser.add_argument('--render_depth', action="store_true", help='Whether we render depth')
parser.add_argument('--render_sfn', action="store_true", help='Whether we render depth')
sep_idx = sys.argv.index("--")
args = parser.parse_args(sys.argv[(sep_idx+1):])
if args.seed is not None:
    np.random.seed(args.seed)

DEBUG = False

VIEWS = args.views
RESOLUTION = 800
R_NEAR = args.near
R_FAR = args.far

RESULTS_PATH = 'nerf_dataset/%s_fixdist_nv%d_dist%s-%s-%s' \
            % (args.name, VIEWS, R_NEAR, R_FAR, args.n_smp)
if args.render_depth:
    RESULTS_PATH += "_depth"
if args.render_sfn:
    RESULTS_PATH += "_sfn"
os.makedirs(RESULTS_PATH, exist_ok=True)
if args.test_only:
    split_to_nviews = {
        "test": args.views,
    }
else:
    split_to_nviews = {
        "train": args.views * 2,
        "test": args.views,
        "val": args.views
    }
fp = bpy.path.abspath(f"//{RESULTS_PATH}")
if not os.path.exists(fp):
    os.makedirs(fp)

# for r in sample_ratios:
#     os.makedirs(os.path.join(RESULTS_PATH, "dist_%s" % r),
#                 exist_ok=True)

MAX_DEPTH = 8.
DEPTH_SCALE =  1. / MAX_DEPTH
COLOR_DEPTH = 8
FORMAT = 'PNG'
UPPER_VIEWS = True


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
#bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if not DEBUG:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
      links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
      # Remap as other types can not represent the full range of depth.
      map = tree.nodes.new(type="CompositorNodeMapValue")
      # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
      map.offset = [0.]
      map.size = [DEPTH_SCALE]
      map.use_min = True
      map.min = [0]
      map.use_max = True
      map.max = [255]
      links.new(render_layers.outputs['Depth'], map.inputs[0])
      links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

stepsize = 360.0 / VIEWS
rotation_mode = 'XYZ'

if not DEBUG:
    for output_node in [depth_file_output, normal_file_output]:
        output_node.base_path = ''


R_delta = (R_FAR - R_NEAR) / float(args.n_smp - 1.)
sample_ratios = [R_NEAR + R_delta * i for i in range(args.n_smp)]
for split, nviews in split_to_nviews.items():

    # One transform file per ratio
    out_data = {smp_r: {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
        "frames": []} for smp_r in sample_ratios}

    for i in range(0, nviews):
        if UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
            # b_empty.rotation_euler = rot
        else:
            #b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
            rot = np.random.uniform(0, 2*np.pi, size=3)

        # Data to store in JSON file
        for smp_r in sample_ratios:
            cam = scene.objects['Camera']
            cam.location = (0, 4.0 * smp_r, 0.5 * smp_r)
            cam_constraint = cam.constraints.new(type='TRACK_TO')
            cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
            cam_constraint.up_axis = 'UP_Y'
            b_empty = parent_obj_to_camera(cam)
            cam_constraint.target = b_empty
            rfpath = 'radius_{}_{}/r_'.format(smp_r, split) + str(i)
            b_empty.rotation_euler = rot
            scene.render.filepath = fp + "/" + rfpath

            depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
            normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

            if args.render_depth:
                depth_file_output.file_slots[0].path = RESULTS_PATH + "/" + rfpath + "_depth_"
            if args.render_sfn:
                normal_file_output.file_slots[0].path = RESULTS_PATH + "/" + rfpath + "_normal_"

            bpy.ops.render.render(write_still=True)  # render still
            frame_data = {
                'full_file_path': scene.render.filepath,
                'file_path': rfpath,
                'transform_matrix': listify_matrix(cam.matrix_world),
                'dist_ratio': smp_r
            }
            if args.render_depth:
                frame_data.update({
                    "full_depth_file_path": bpy.path.abspath(
                        depth_file_output.file_slots[0].path),
                    "depth_file_path": rfpath + "_depth_",
                    "depth_scale": DEPTH_SCALE,
                    "max_depth": MAX_DEPTH
                })

            if args.render_sfn:
                frame_data.update({
                    'full_normal_file_path': bpy.path.abspath(
                        normal_file_output.file_slots[0].path),
                    "normal_file_path": rfpath + "_normal_"
                })
            out_data[smp_r]['frames'].append(frame_data)


    if not DEBUG:
        for smp_r, smp_r_out_data in out_data.items():
            with open(fp + '/' + 'transforms_radius%s_%s.json' % (smp_r, split), 'w') as out_file:
                json.dump(smp_r_out_data, out_file, indent=4)

