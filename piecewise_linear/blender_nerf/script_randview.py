# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import argparse, sys, os
import json
import numpy as np
import argparse
sep_idx = sys.argv.index("--")

parser = argparse.ArgumentParser(description='NeRF training set.')
parser.add_argument('name', type=str, help='Name')
parser.add_argument('views', type=int, help='Number of views')
parser.add_argument('near', type=float, help='Distance ratio (near)')
parser.add_argument('far', type=float, help='Distance ratio (far)')

args = parser.parse_args(sys.argv[(sep_idx+1):])

import bpy


DEBUG = False

VIEWS = {
    "train": 2 * args.views,
    "val": args.views,
    "test": args.views,
}
RESOLUTION = 800
DISTANCE_RATIO_NEAR = args.near
DISTANCE_RATIO_FAR = args.far
RESULTS_PATH = 'nerf_dataset/%s_randdist_nv%d_dist%s-%s' \
            % (args.name, args.views, DISTANCE_RATIO_NEAR, DISTANCE_RATIO_FAR)
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
RANDOM_VIEWS = True
UPPER_VIEWS = True


fp = bpy.path.abspath(f"//{RESULTS_PATH}")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

if not os.path.exists(fp):
    os.makedirs(fp)

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
      map.offset = [-0.7]
      map.size = [DEPTH_SCALE]
      map.use_min = True
      map.min = [0]
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

rotation_mode = 'XYZ'

if not DEBUG:
    for output_node in [depth_file_output, normal_file_output]:
        output_node.base_path = ''


# for i in range(0, VIEWS):
for split, views in VIEWS.items():
    stepsize = 360.0 / views
    # Data to store in JSON file
    out_data = {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    }
    out_data['frames'] = []
    for i in range(0, views):
        cam = scene.objects['Camera']
        dist_ratio = (
            np.random.uniform(0, 1) * (DISTANCE_RATIO_FAR - DISTANCE_RATIO_NEAR) +
            DISTANCE_RATIO_NEAR)
        cam.location = (0, 4.0 * dist_ratio, 0.5 * dist_ratio)
        cam_constraint = cam.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        b_empty = parent_obj_to_camera(cam)
        cam_constraint.target = b_empty
        if RANDOM_VIEWS:
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
            rfpath = '{}/r_{}'.format(split, i)
        else:
            print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
            rfpath =  '{}/r_{}_{0:03d}'.format(split, i, int(i * stepsize))
        scene.render.filepath = fp + "/" + rfpath

        # depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
        # normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

        if DEBUG:
            break
        else:
            bpy.ops.render.render(write_still=True)  # render still
        frame_data = {
            'full_file_path': scene.render.filepath,
            'file_path': rfpath,
            'rotation': radians(stepsize),
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)

        if RANDOM_VIEWS:
            if UPPER_VIEWS:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            b_empty.rotation_euler[2] += radians(stepsize)

    if not DEBUG:
        with open(fp + '/' + '{}_transforms.json'.format(split), 'w') as out_file:
            json.dump(out_data, out_file, indent=4)

