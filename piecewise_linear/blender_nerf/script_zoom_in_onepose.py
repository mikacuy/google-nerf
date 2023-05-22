# A simple script that uses blender to render views of a single object by
# Zooming in the camera.
# Also produces depth map at the same time.

from math import radians
import bpy
import argparse, sys, os
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='NeRF training set.')
parser.add_argument('name', type=str, help='Name')
parser.add_argument('n_smp', type=int, help='Number of different distance samples')
parser.add_argument('near', type=float, help='Distance ratio (near)')
parser.add_argument('far', type=float, help='Distance ratio (far)')
parser.add_argument("--center", nargs=3, type=float,
                    help="Frame center (x,y,z, coordinate).")
parser.add_argument("--rotate", nargs=3, type=float, help="Rotation (x,y,z, euler rotation).")
parser.add_argument("--resolution", type=int, default=800, help="Resolution.")
parser.add_argument('--seed', type=int, default=666, help='RAndom seed.')
sep_idx = sys.argv.index("--")
args = parser.parse_args(sys.argv[(sep_idx+1):])
np.random.seed(args.seed)


DEBUG = False

RESOLUTION = 800
R_NEAR = args.near
R_FAR = args.far
R_delta = (R_FAR - R_NEAR) / float(args.n_smp - 1.)
sample_ratios = [R_NEAR + R_delta * i for i in range(args.n_smp)]
RESULTS_PATH = 'nerf_dataset/%s_zoomin_nsmp%d_dist%s-%s_C%s-%s-%s_R%s-%s-%s' \
            % (args.name, args.n_smp, R_NEAR, R_FAR,
               args.center[0], args.center[1], args.center[2],
               args.rotate[0], args.rotate[1], args.rotate[2])
os.makedirs(RESULTS_PATH, exist_ok=True)
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
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
    origin = tuple(args.center)
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


rotation_mode = 'XYZ'

if not DEBUG:
    for output_node in [depth_file_output, normal_file_output]:
        output_node.base_path = ''

rot = np.array(args.rotate)

out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    'frames': []
}

for r_idx, smp_r in enumerate(sample_ratios):
    cam = scene.objects['Camera']
    cam.location = (0, 4.0 * smp_r, 0.5 * smp_r)
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    rfpath = 'r_{}'.format(r_idx)
    b_empty.rotation_euler = rot
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
        'transform_matrix': listify_matrix(cam.matrix_world),
        'dist_ratio': smp_r
    }
    out_data['frames'].append(frame_data)

if not DEBUG:
    with open(fp + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
