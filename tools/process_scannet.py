import os 
import glob 
import os.path as osp 
from plyfile import PlyData
import numpy as np 
from plyfile import PlyData,PlyElement
from tqdm import tqdm
import trimesh
import jittor as jt
from jmesh.config.constant import SCANNET_CLASS_REMAP
from jmesh.utils.general import multi_process
from jmesh.utils.data_utils import vertex2face_labels, vertex2vertex_map, voxelize_mesh
from jmesh.layers.face_pool import build_mesh_level

def crop_mesh(vertex_data,faces,save_dir,name,face_num=6000,range_ratio=[0.95,1.05],with_labels=True):
    face_num = min(face_num,len(faces))
    xyz = vertex_data[:,:3]
    center_face = xyz[faces].mean(axis=1)
    max_radius = (center_face.max(axis=0)-center_face.min(axis=0)+0.2).max()

    used = np.ones((faces.shape[0],),dtype=np.int32)
    part = 1
    all_face = 0
    while used.sum()>0:
        # print("crop",name)
        index, = np.where(used==1)
        center_index = index[0]
        center = center_face[center_index]
        l_radius = 0
        r_radius = max_radius
        radius = (l_radius+r_radius)/2

        while l_radius<r_radius:
            # print("radius",radius)
            radius = (l_radius+r_radius)/2
            dis = np.linalg.norm(center_face-center,axis=1)
            valid = dis<=radius
            s = valid.sum()
            # print("split:",part,l_radius,r_radius,s,face_num)

            if s>range_ratio[1]*face_num:
                r_radius = radius-0.01
            elif s<range_ratio[0]*face_num:
                l_radius = radius+0.01
            else:
                break
        dis = np.linalg.norm(vertex_data[:,:3]-center,axis=1)
        valid = dis<=radius
            
        valid2 = valid[faces].sum(axis=1) == 3
        valid_faces = faces[valid2]
        valid = np.zeros(len(valid), dtype=bool)
        valid[valid_faces] = True
        valid_vertexes = vertex_data[valid]


        index_map = np.cumsum(valid)-1
        valid_faces = index_map[valid_faces]
        
        if with_labels:
            vertex_labels = valid_vertexes[:,-1].astype(np.int32)
            face_labels = vertex2face_labels(valid_faces,vertex_labels)
        else:
            face_labels = None
            vertex_labels = None 

        save_file = osp.join(save_dir,f"{name}_{part}")
        vertices = valid_vertexes[:,:3]
        colors = valid_vertexes[:,3:6]
        
        mesh = trimesh.Trimesh(vertices=vertices,faces=valid_faces,vertex_colors=colors,process=False,maintain_order=True)
        mesh.export(save_file+".obj")
        
        if with_labels:
            np.savetxt(save_file+"_vertex_labels.txt",vertex_labels,fmt="%d")
            np.savetxt(save_file+"_face_labels.txt",face_labels,fmt="%d")
        
        levels = build_mesh_level(valid_faces,level=6)
        jt.save([valid,valid2,center,radius],save_file+"_info.pkl")
        jt.save([levels,None,face_labels,None,None],save_file+".pkl")

        used[valid2]=0
        part +=1
        all_face+=valid2.sum()

def read_scannet_ply(mesh_file,with_mapping=True):
    mesh_data = PlyData.read(mesh_file)
    vertex_data = [np.asarray(mesh_data['vertex'][x]) for x in ['x','y','z','red','green','blue']]
    
    labels_file = mesh_file.replace('.ply', '.labels.ply')
    with_labels = False
    if osp.exists(labels_file):
        label_data = PlyData.read(labels_file)
        vertex_labels = np.asarray(label_data['vertex']['label'])
        if with_mapping:
            # FIX: THREE MESHES HAVE CORRUPTED LABEL IDS,from dcm-net
            vertex_labels[vertex_labels>40] = 0
            vertex_labels = SCANNET_CLASS_REMAP[vertex_labels]
        vertex_data = vertex_data+[vertex_labels]
        with_labels = True

    vertex_data = np.stack(vertex_data,axis=1)
    
    faces = np.stack([face[0] for face in mesh_data['face']],axis=0)
    return vertex_data,faces,with_labels

def voxelize_mesh_wrapper(args):
    read_func,mesh_file,save_dir,name,voxel_size,face_num = args 
    vertex_data,faces,with_labels = read_func(mesh_file)

    if with_labels:
        vertex_labels = vertex_data[:,-1].astype(np.int32)
        vertex_data = vertex_data[:,:6]

    orig_vertices = vertex_data[:,:3]
    vertex_data, faces, idx = voxelize_mesh(vertex_data, faces, voxel_size)

    root_save_dir = "/".join(save_dir.split("/")[:-1])
    task = save_dir.split("/")[-1]
    jt.save((idx,vertex_data[:,:3],faces,vertex_labels if with_labels else None),os.path.join(root_save_dir,"raw_map",task,name+".pkl"))
    
    
    if with_labels:
        vertex_labels = vertex2vertex_map(orig_vertices,vertex_data[:,:3],vertex_labels)
        vertex_data = np.concatenate([vertex_data,vertex_labels[:,None]],axis=1)
    crop_mesh(vertex_data,faces,save_dir,name,with_labels=with_labels,face_num=face_num)

def preprocess_voxel():
    processes = 64 
    voxel_size = 2
    face_num = 100000
    
    data_dir = "/mnt/disk/lxl/datasets/scannet"

    assert os.path.exists(data_dir), "The data dir must exist."
    save_dir = f"datasets/scannet/scannet_voxel_{voxel_size}_split{face_num}"
    # if osp.exists(save_dir):
    #     shutil.rmtree(save_dir)
    tasks = ["train","val","test"]
    for task in tasks:

        with open(osp.join(f"datasets/scannet/meta/scannetv2_{task}.txt"),"r") as f:
            rooms = f.read().splitlines()
        
        if task == "test":
            sub_dir = "scans_test"
        else:
            sub_dir = "scans"

        files = sorted([x for x in glob.glob(f"{data_dir}/{sub_dir}/*/*clean_2.ply")
                          if x.split('/')[-1].rsplit('_', 3)[0] in rooms])
        
        task_save_dir = osp.join(save_dir,task)
        os.makedirs(osp.join(save_dir,"raw_map",task),exist_ok=True)
        os.makedirs(task_save_dir,exist_ok=True)
        files = [(read_scannet_ply,mesh_file,task_save_dir,mesh_file.split("/")[-2],voxel_size,face_num) for  mesh_file in files]
        multi_process(voxelize_mesh_wrapper,files,processes=processes)

if __name__ == "__main__":
    preprocess_voxel()