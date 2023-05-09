import trimesh
import os.path as osp 
import os 

import numpy as np 
import jittor as jt 
from jmesh.layers.face_pool import build_mesh_level

from sklearn.neighbors import BallTree

def simplify_mesh(vertex_data,faces,face_num):
    from jmesh.utils import jmesh_simplify as jms 
    mesh = jms.TriangleMesh(
            vertices=vertex_data[:,:3],
            triangles=faces)
    mesh.vertex_colors = vertex_data[:,3:6]
    vertex_map = mesh.simplify_quadric_decimation(face_num)
    vertex_map = np.array(vertex_map)
    faces = np.array(mesh.triangles)
    vertices = np.array(mesh.vertices)
    vertex_colors = np.array(mesh.vertex_colors)
    return vertices, vertex_colors,vertex_map,faces


def vertex2face_labels(faces,vertex_labels):
    face_labels = np.sort(vertex_labels[faces],1)[:,1]
    return face_labels

def face2vertex_labels(VN,faces,face_labels):
    vertex_labels = np.zeros((VN,22),dtype=np.int32)
    for i in range(len(faces)):
        for j in range(3):
            vertex_labels[faces[i,j],face_labels[i]]+=1
    vertex_labels = np.argmax(vertex_labels,1)
    return vertex_labels


def vertex2vertex_map(orig_vertices,now_vertices,map_data):
    ball_tree = BallTree(orig_vertices)

    _, ind = ball_tree.query(now_vertices, k=1)
    map_data = map_data[ind.flatten()]
    return map_data



def mesh2mesh_map_v0(mesh1,mesh2):
    center1 = np.array(mesh1.vertices)[np.array(mesh1.faces)].mean(axis=1)
    center2 = np.array(mesh2.vertices)[np.array(mesh2.faces)].mean(axis=1)
    ball_tree = BallTree(center1)
    _, ind = ball_tree.query(center2, k=1)
    
    return ind.flatten()

def sampling_point(mesh,pointnum):
    face_vertices = np.array(mesh.vertices)[np.array(mesh.faces)]
    areas = np.array(mesh.area_faces)
    perpoint_area = np.sum(areas)/pointnum
    points = []
    indexes = []
    for index,(f_vs,a) in enumerate(zip(face_vertices,areas)):
        n = int(np.ceil(a/perpoint_area))+1
        tmp_vs = [f_vs]
        while n>0:
            new_tmp_vs = []
            for vs in tmp_vs:
                center = (vs[0]+vs[1]+vs[2])/3
                points.append(center)
                indexes.append(index)
                c01 = (vs[0]+vs[1])/2
                c02 = (vs[0]+vs[2])/2
                c12 = (vs[1]+vs[2])/2
                new_tmp_vs.append((vs[0],c02,c01))
                new_tmp_vs.append((vs[1],c12,c01))
                new_tmp_vs.append((vs[2],c02,c12))
                new_tmp_vs.append((c01,c02,c12))
                n-=1
            tmp_vs = new_tmp_vs
    points = np.array(points)
    indexes = np.array(indexes)
    return points,indexes

def mesh2mesh_map(mesh1,mesh2):
    points,indexes = sampling_point(mesh1,10000)
    center2 = np.array(mesh2.vertices)[np.array(mesh2.faces)].mean(axis=1)
    ball_tree = BallTree(points)
    _, ind = ball_tree.query(center2, k=1)
    return indexes[ind.flatten()]


def voxelize_mesh(vertex, face, voxel_size):
    assert voxel_size>0

    pos_xyz = vertex[:,:3] - vertex[:,:3].min(axis=0)
    tmp_ids = np.floor(pos_xyz/(voxel_size/100)) # centimeter voxel_size to meter
    # Mags = np.ceil(np.log(np.max(tmp_ids,axis=0))/np.log(10.))

    # print(Mags)
    # print(tmp_ids.max(0))

    ids = tmp_ids.astype(np.int32)
    dims = ids.max(0)+1
    # Mags = Mags.astype(np.int32)

    ids_1D = ids[:,0]*dims[1]*dims[2] + ids[:,1]*dims[2] + ids[:,2]
    
    id_unique, idx, counts = np.unique(ids_1D,return_counts=True,return_inverse=True)

    vertex = jt.array(vertex)
    imap = jt.array(idx)
    N,M = vertex.shape
    New_N = len(id_unique)

    vertex = vertex.reindex_reduce(
                    op="sum",
                    shape=[New_N,M],
                    indexes=[
                        '@e0(i0)',
                        'i1'
                    ],
                    extras=[imap,])
    new_vertex = vertex.numpy()/counts[:,None]
    
    new_face = idx[face]
    non_degenerate = ~((new_face[:,0]==new_face[:,1]) | (new_face[:,2]==new_face[:,1]) | (new_face[:,0]==new_face[:,2]))
    new_face = new_face[non_degenerate,:]
    return new_vertex, new_face, idx
