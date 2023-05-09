from jittor.dataset import Dataset
import jittor as jt 
import numpy as np 
import os 
import glob 
import json
from tqdm import tqdm
import trimesh
from pathlib import Path
import os.path as osp
from jmesh.utils.registry import DATASETS
from jmesh.data.tensor import MeshTensor
from jmesh.utils.general import summary_size, to_jt_var
from .utils import Compose
from jmesh.config import get_cfg

def load_feature(mesh,feats,color_aug):
    F = mesh.faces
    V = mesh.vertices
    VN = mesh.vertex_normals
    FN = mesh.face_normals
    colors = mesh.visual.face_colors
    area = mesh.area_faces[:,None]
    Fs = mesh.faces.shape[0]

    colors = np.array(colors)/255.
    colors = colors[:,:3]
    if color_aug:
        colors += np.random.randn(3)*0.1

    if get_cfg().flip_normal:
        FN = FN* (((np.random.randn(FN.shape[0])>0)-1)[:,None])
    
    FV =  V[F.flatten()].reshape(-1,3,3)
    height = FV[:,:,2]
    lengths = np.linalg.norm(FV-FV[:,[1,2,0],:],axis=-1)
    center = FV.mean(axis=1)
    curvs = np.hstack([
        (VN[F[:, 0]] * FN).sum(axis=1,keepdims=True),
        (VN[F[:, 1]] * FN).sum(axis=1,keepdims=True),
        (VN[F[:, 2]] * FN).sum(axis=1,keepdims=True),
    ])
    curvs = np.sort(curvs,axis=1)
    angles = np.sort(mesh.face_angles, axis=1)

    features = []
    for i,j in zip(["area","normal","center","angle","curvs",'color',"length","height"],[area,FN,center,angles,curvs,colors,lengths,height]):
        if i in feats:
            features.append(j)

    feature = np.concatenate(features,axis=1)
    feature = feature.T
    return feature,Fs,np.asarray(center)

@DATASETS.register_module()
class Scannet(Dataset):
    classes = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain',
               'toilet', 'sink', 'bathtub', 'otherfurniture']

    def __init__(self, dataroot, 
                       transforms=None,
                       batch_size=1, 
                       mode="train", 
                       shuffle=False, 
                       pattern = "scene*.obj",
                       num_workers=0,
                       level=6,
                       file_ext=".off",
                       color_aug = True,
                       feats = ["area","normal","center","angle","curvs","color"],
                       drop_last=False):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,drop_last=drop_last)
        assert mode in ["train","val","test","trainval"]
        self.transforms = Compose(transforms)
        self.color_aug = color_aug
        
        self.files = self.browse_dataroot(dataroot,mode=mode,pattern=pattern)
        self.level = level

        self.feats = feats
        self.mode = mode 
        self.total_len=len(self.files)
        self.use_max = True

    def browse_dataroot(self,dataroot,mode,pattern):
        name = dataroot.split("/")[-1]
        mesh_paths = []
        for mesh_file in glob.glob(osp.join(dataroot,mode,pattern)):
            mesh_paths.append(mesh_file)
            
            # if obj_path.suffix == '.obj':
            #     # mesh = trimesh.load_mesh(obj_path, process=False)
            #     # if len(mesh.faces)>15000:
            #     #     continue
            #     mesh_paths.append(str(obj_path))
        
        # mesh_paths = mesh_paths[:200]
        # mesh_paths = mesh_paths*5
        # if mode == "train":
        #     mesh_paths = mesh_paths*100
        #     pass 
        # print(len(mesh_paths),len(set(mesh_paths)))
        if os.path.exists(f"tmp/scannet_{name}_{mode}.pkl"):
            mesh_paths = jt.load(f"tmp/scannet_{name}_{mode}.pkl")
            return mesh_paths
        if mode in ["train","val","test","trainval"]:
            new_mesh_paths = []
            for mesh_file in tqdm(mesh_paths):
                try:
                    mesh = trimesh.load(mesh_file,process=False,maintain_order=True)
                    
                    levels,indexes,labels,_,_ = jt.load(mesh_file.replace(".obj",".pkl"))
                    if mode == "test" or (len(mesh.faces)==len(labels) and (labels>0).sum().item()>500):
                        new_mesh_paths.append(mesh_file)
                    else:
                        print(mesh_file,len(mesh.faces),len(labels))
                except:
                    print("Error",mesh_file)
            mesh_paths = new_mesh_paths
            os.makedirs("tmp",exist_ok=True)

            jt.save(mesh_paths,f"tmp/scannet_{name}_{mode}.pkl")
                
        return mesh_paths
    

    def __getitem__(self, idx):
        # while True:
        #     mesh_file = self.files[idx]
        #     try:
        #         mesh = trimesh.load(mesh_file,process=False,maintain_order=True)
        #         # labels = np.loadtxt(mesh_file.replace(".obj","_face_labels.txt"),dtype=np.int32)
        #         # levels = build_mesh_level(mesh.faces,level=self.level)
        #         levels,indexes,labels = jt.load(mesh_file.replace(".obj",".pkl"))
        #         if len(mesh.faces)==len(labels) and (labels>0).sum().item()>500:
        #             break
        #     except Exception as e:
        #         print("Error:",mesh_file,e) 
        #     # print("Change:",mesh_file)               
        #     idx = np.random.choice(np.arange(self.total_len))
        mesh_file = self.files[idx]
        mesh = trimesh.load(mesh_file,process=False,maintain_order=True)
        levels,indexes,labels,knn_dis,knn_index = jt.load(mesh_file.replace(".obj",".pkl"))
        if labels is None:
            labels = np.zeros((len(mesh.faces),),dtype=np.int32)
            
        mesh = self.transforms(mesh)
        feature,Fs,center = load_feature(mesh,self.feats,self.color_aug and self.mode=="train")
        assert Fs == len(labels),f"{mesh_file},{Fs},{len(labels)}"

        return feature,Fs,levels,labels,center,mesh_file

    def collate_batch(self, batch):
        feats,Fs,levels,labels,centers,mesh_files = zip(*batch)
        N = len(batch)        
        max_f = max(Fs)
        if self.use_max == False:
            # src 12600 
            max_f = 105000
            # max_f = 12600

        np_feats = np.zeros((N, feats[0].shape[0], max_f), dtype=np.float32)
        np_labels = np.zeros((N, max_f), dtype=np.int32)
        for i in range(N):
            np_feats[i, :, :Fs[i]] = feats[i]
            np_labels[i, :Fs[i]] = labels[i]
        np_levels = self.collate_levels(levels,centers)
        feats,levels,labels = to_jt_var([np_feats,np_levels,np_labels])


        meshtensor = MeshTensor(feats=feats,
                                  level=0,
                                  levels=levels,
                                  mesh_files = mesh_files)
        self.use_max = True
        return meshtensor, labels


    def collate_levels(self,levels,centers):
        levels = list(zip(*levels))
        L = len(levels)
        N = len(levels[0])
        Fs = []
        for i in range(L):
            F = []
            for j in range(N):
                F.append(len(levels[i][j][0]))
            Fs.append(np.array(F,dtype=np.int32))
        np_levels = []
        # to avoid jittor allocate too much memory.
        static_FS = [105000,  61174,  37955,  25030,  17597,  13387]
        for i in range(L):
            max_f = Fs[i].max().item()
            if self.use_max == False:
                max_f = static_FS[i]
            np_pool_mask = np.zeros((N,max_f),dtype=np.int32)
            np_adj = -np.ones((N,max_f,3),dtype=np.int32)
            np_center = []
            new_centers = []
            for j,(pool_mask,adj) in enumerate(levels[i]):
                np_pool_mask[j,:Fs[i][j]] = pool_mask
                np_adj[j,:Fs[i][j],:] = adj
                np_center.append(centers[j])
                n_c = centers[j][pool_mask==1]
                new_centers.append(n_c)
            centers = new_centers
            np_indexes = (np_pool_mask.cumsum(1)*np_pool_mask)-1
            np_levels.append((np_pool_mask,np_adj,Fs[i],np_center,np_indexes))#,np_knn_adj,np_knn_dis))
        return np_levels
