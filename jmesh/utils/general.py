import jittor as jt 
import numpy as np 
import os
import time 
import sys
import random
import glob
import shutil
import warnings
import pickle
from multiprocessing import Pool
from tqdm import tqdm


def current_time():
    return time.asctime( time.localtime(time.time()))

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.seed(seed)

def build_file(work_dir,prefix):
    """ build file and makedirs the file parent path """
    work_dir = os.path.abspath(work_dir)
    prefixes = prefix.split("/")
    file_name = prefixes[-1]
    prefix = "/".join(prefixes[:-1])
    if len(prefix)>0:
        work_dir = os.path.join(work_dir,prefix)
    os.makedirs(work_dir,exist_ok=True)
    file = os.path.join(work_dir,file_name)
    return file 

def to_jt_var(data):
    """
        convert data to jt_array
    """
    def _to_jt_var(data):
        if isinstance(data,(list,tuple)):
            data =  [_to_jt_var(d) for d in data]
        elif isinstance(data,dict):
            data = {k:_to_jt_var(d) for k,d in data.items()}
        elif isinstance(data,np.ndarray):
            data = jt.array(data)
        elif data is None:
            return data
        elif not isinstance(data,(int,float,str,np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        return data
    
    return _to_jt_var(data) 

def sync(data,reduce_mode="mean",to_numpy=True):
    """
        sync data and convert data to numpy
    """
    def _sync(data):
        if isinstance(data,(list,tuple)):
            data =  [_sync(d) for d in data]
        elif isinstance(data,dict):
            data = {k:_sync(d) for k,d in data.items()}
        elif isinstance(data,jt.Var):
            if jt.in_mpi:
                data = data.mpi_all_reduce(reduce_mode)
            if to_numpy:
                data = data.numpy()
        elif not isinstance(data,(int,float,str,np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        return data
    
    return _sync(data) 


def check_file(file,ext=None):
    if file is None:
        return False
    if not os.path.exists(file):
        warnings.warn(f"{file} is not exists")
        return False
    if not os.path.isfile(file):
        warnings.warn(f"{file} must be a file")
        return False
    if ext:
        if not os.path.splitext(file)[1] in ext:
            # warnings.warn(f"the type of {file} must be in {ext}")
            return False
    return True

def search_ckpt(work_dir):
    files = glob.glob(os.path.join(work_dir,"checkpoints/ckpt_*.pkl"))
    if len(files)==0:
        return None
    files = sorted(files,key=lambda x:int(x.split("_")[-1].split(".pkl")[0]))
    return files[-1]  

def clean(work_dir):
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        
def multi_process(func,files,processes=1):
    if processes <=1:
        for mesh_file in tqdm(files):
            func(mesh_file)
    else:
        with Pool(processes=processes) as pool:
            r = list(tqdm(pool.imap(func, files), total=len(files)))


def save(data,file):
    with open(file,"wb") as f:
        pickle.dump(data,f)

def load(file):
    data = None
    with open(file,"rb") as f:
        data = pickle.load(f)
    return data

def summary_size(data):
    def _summary_size(data):
        if isinstance(data,(list,tuple)):
            data =  [_summary_size(d) for d in data]
        elif isinstance(data,dict):
            data = {k:_summary_size(d) for k,d in data.items()}
        elif not isinstance(data,(int,float,str,np.ndarray)):
            raise ValueError(f"{type(data)} is not supported")
        else:
            data = [sys.getsizeof(data)/1e6]
        return sum(data)
    size = _summary_size(data) 
    # size /=10^9
    return size

def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print(num_params)
    print('-----------------------------------------------')