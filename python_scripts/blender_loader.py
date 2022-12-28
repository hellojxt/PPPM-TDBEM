import os
import bpy
import bmesh
from glob import glob
import numpy as np


class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    vertex = list(map(float, line[1:].split()))
                    self.vertices.append(vertex)
                elif line[0] == "f":
                    line = line[1:].split()
                    face = []
                    for item in line:
                        if item.find('/') > 0:
                            item = item[:item.find('/')]
                        face.append(int(item)-1)
                    self.faces.append(face)
            f.close()
        except IOError:
            print(f'{fileName} not found.')

        self.vertices = np.asarray(self.vertices)
        self.faces = np.asarray(self.faces)


def load_base_mesh(scene, base_mesh_name):
    old_objs = set(scene.objects)
    fpath = os.path.join(ROOT_DIR, f'{PREFIX}_{scene.frame_start}.obj')
    bpy.ops.import_scene.obj(filepath=fpath, split_mode='OFF')
    obj = list(set(scene.objects) - old_objs)[0]
    obj.name = base_mesh_name
    return obj


def load_shape_keys(scene, base_obj):
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    bpy.ops.object.shape_key_add()
    if (bpy.context.object.mode == 'OBJECT'):
        bpy.ops.object.mode_set(mode="EDIT")
    for f in range(scene.frame_start + 1, MAX_FRAME + 1):
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.shape_key_add()
        bpy.ops.object.mode_set(mode="EDIT")
        bm = bmesh.from_edit_mesh(base_obj.data)
        obj_mesh = ObjLoader(os.path.join(ROOT_DIR, f'{PREFIX}_{f}.obj'))
        for v in bm.verts:
            v.co = obj_mesh.vertices[v.index]
        bmesh.update_edit_mesh(base_obj.data)
        bm.free()
    bpy.ops.object.mode_set(mode="OBJECT")


def load_key_frames(scene, base_obj):
    bpy.ops.object.select_all(action='DESELECT')
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    key_frame_num = len(base_obj.data.shape_keys.key_blocks)
    for i in range(key_frame_num):
        frame = scene.frame_start + i
        shape_key = base_obj.data.shape_keys.key_blocks[i]
        if i > 0:
            shape_key.value = 0
            shape_key.keyframe_insert(data_path='value', frame=frame - 1)
        shape_key.value = 1
        shape_key.keyframe_insert(data_path='value', frame=frame)
        if i < key_frame_num - 1:
            shape_key.value = 0
            shape_key.keyframe_insert(data_path='value', frame=frame + 1)


if __name__ == '__main__':
    ROOT_DIR = '/home/jxt/PPPM-TDBEM/experiments/rigidbody/output/spolling_bowl/ghost/mesh'
    file_list = glob(os.path.join(ROOT_DIR, '*.obj')) 
    PREFIX = os.path.basename(file_list[0]).split('_')[0]
    file_list.sort(key=lambda x: int(
        os.path.basename(x).split('_')[1].split('.')[0]))
    MAX_FRAME = int(os.path.basename(
        file_list[-1]).split('_')[1].split('.')[0])
    scene = bpy.context.scene
    obj = load_base_mesh(scene, PREFIX)
    load_shape_keys(scene, obj)
    load_key_frames(scene, obj)
