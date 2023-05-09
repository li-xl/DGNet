import jittor as jt 
import numpy as np 
from tqdm import tqdm
import os.path as osp 
import os
from jmesh.config.constant import SCANNET_MAP,SCANNET_CLASS_REMAP
from jmesh.utils.data_utils import face2vertex_labels

def merge_results(result_path):
    results = jt.load(result_path)
    m_results = {}
    for mesh_file,output,target in tqdm(results):
        scene_name = '_'.join(mesh_file.split("/")[-1].split("_")[:2])
        valid,valid2,center,radius = jt.load(mesh_file.replace(".obj","_info.pkl"))
        num = valid2.sum()
        assert target[num:].sum() == 0
        output = output[:,:num]
        if scene_name not in m_results:
            m_results[scene_name] = np.zeros((output.shape[0],len(valid2)),dtype=np.float32)
        m_results[scene_name][:,valid2] += output

    m_results = {k:d.argmax(axis=0) for k,d in m_results.items()}
    return m_results

def map_raw_results(m_results,data_dir,save_dir,mode="val"):
    num_classes = len(SCANNET_MAP)
    conf_matrix = np.zeros((num_classes,num_classes))
    os.makedirs(save_dir,exist_ok=True)
    for scene_name,pred in tqdm(m_results.items()):
        data_file = osp.join(data_dir,"raw_map",mode,scene_name+".pkl")
        idx,vertices,faces,gt_labels = jt.load(data_file)
        VN = idx.max()+1
        pred_v = face2vertex_labels(VN,faces,pred)
        
        pred_v = pred_v[idx]
        raw_pred_v = SCANNET_MAP[pred_v]
        save_file = osp.join(save_dir,scene_name+".txt")
        np.savetxt(save_file,raw_pred_v,fmt="%d")
        if gt_labels is None:
            continue
        # hack for bincounting 2 arrays together
        x = pred_v + num_classes * gt_labels
        bincount_2d = np.bincount(x.astype(np.int32), minlength=num_classes**2)
        assert bincount_2d.size == num_classes**2
        conf_matrix += bincount_2d.reshape((num_classes, num_classes))
    
    # remove unknown
    conf_matrix[0,:]=0
    conf_matrix[:,0]=0
    
    true_positive = np.diag(conf_matrix)
    false_positive = np.sum(conf_matrix, 0) - true_positive
    false_negative = np.sum(conf_matrix, 1) - true_positive

    iou = true_positive / np.maximum(1,(true_positive + false_positive + false_negative))
    precision = true_positive / np.maximum(1,(true_positive + false_negative))

    opre = np.sum(true_positive) / np.maximum(1,np.sum(conf_matrix))

    miou = np.mean(iou[1:])
    mpre = np.mean(precision[1:])
    print(miou,mpre,opre)

        
def main():
    result_path = "work_dirs/scene_scannetv2_e100_32_val/val/200.pkl"
    save_dir = "results/scannet_voxel_2_100000_val"
    mode = "val"
    
    data_dir = "datasets/scannet/scannet_voxel_2_split100000"
    m_results = merge_results(result_path=result_path)
    map_raw_results(m_results,data_dir,save_dir,mode=mode)

if __name__ == "__main__":
    main()

    