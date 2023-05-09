from jmesh.config.constant import SCANNET_REMAP_COLOR,SCANNET_COLOR
import trimesh
import numpy as np


def visualize(mesh_file,save_file,vertex_labels=None,face_labels=None,use_remap=True):
    assert (vertex_labels is None and face_labels is not None) or (vertex_labels is not None and face_labels is None)
    mesh = trimesh.load(mesh_file,process=False,maintain_order=True)
    # print(mesh.vertices)
    # print(np.asarray(mesh.vertices).shape)

    # print(len(mesh.vertices))
    # print(len(mesh.faces))
    # print(np.array(mesh.faces).max())
    assert face_labels is None or len(mesh.faces) == len(face_labels)
    assert vertex_labels is None or len(mesh.vertices) == len(vertex_labels)
    
    if use_remap:
        colors = SCANNET_REMAP_COLOR
    else:
        colors = SCANNET_COLOR

    if vertex_labels is not None:
        vertex_colors = colors[vertex_labels]
        mesh.visual.vertex_colors=vertex_colors
    
    if face_labels is not None:
        face_colors = colors[face_labels]
        mesh.visual.face_colors = face_colors
    mesh.export(save_file)

def visualize_seg(faces,vertices,face_labels,save_file):
    face_colors = SCANNET_COLOR[face_labels]
    mesh = trimesh.Trimesh(vertices=vertices,faces=faces,face_colors=face_colors)
    mesh.export(save_file)


