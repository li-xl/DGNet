from tqdm import tqdm 
import numpy as np 
import jittor as jt 
from jittor import nn
import os.path as osp
import os

import jmesh
from jmesh.config.config import get_cfg,save_cfg
from jmesh.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jmesh.utils.general import build_file,current_time,search_ckpt,check_file,clean,print_network
from jmesh.data.metrics import IoU,Accuracy

class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg 

        self.work_dir = cfg.work_dir

        if cfg.clean and jt.rank==0:
            clean(self.work_dir)

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path

        self.model = build_from_cfg(cfg.model,MODELS)
        print_network(self.model)
        self.optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=self.model.parameters())
        self.scheduler = build_from_cfg(cfg.lr_scheduler,SCHEDULERS,optimizer=self.optimizer)

        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi)
        self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
        
        if jt.rank ==0 :
            self.logger = build_from_cfg(self.cfg.logger,HOOKS,work_dir=self.work_dir)
            save_file = build_file(self.work_dir,prefix="config.yaml")
            save_cfg(save_file)
        else:
            self.logger = None

        self.epoch = 0
        self.iter = 0

        self.max_epoch = cfg.max_epoch
        assert cfg.max_epoch is not None,"Must set max epoch in config"

        if self.resume_path is None and not cfg.clean:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()

    def run(self):
        if jt.rank == 0:
            self.logger.print_log("Start running")
        
        while self.epoch < self.max_epoch:            
            self.train()
            self.epoch +=1

            if self.epoch % self.eval_interval == 0:
                self.model.eval()
                self.val()

            if self.epoch % self.checkpoint_interval == 0:
                self.save()

    def train(self):
        # self.logger.print_log("Training....")
        self.model.train()

        if self.cfg.iou_metric:
            metric = IoU(mode="train",ignore_index=self.cfg.ignore_index)
        else:
            metric = Accuracy(mode="train")

        files = []
        for batch_idx,(feats,targets) in tqdm(
                                             enumerate(self.train_dataset),
                                             desc=f'Train {self.epoch}',
                                             total=len(self.train_dataset),
                                             disable=jt.rank!=0):
            outputs = self.model(feats)
            loss = nn.cross_entropy_loss(outputs.unsqueeze(dim=-1), targets.unsqueeze(dim=-1), 
                             ignore_index=self.cfg.ignore_index)
            self.optimizer.step(loss)
            if self.scheduler:
                self.scheduler.step(iters=self.iter,epochs=self.epoch,by_epoch=True)
            
            files.extend(feats.get("mesh_files"))
            if self.iter % 10 == 0:
                jt.sync_all(True)
                jt.gc()
            if self.iter>0 and self.iter % self.log_interval==0:
                data = metric.add(outputs,targets)
                data.update(dict(
                    name = self.cfg.name,
                    lr = self.optimizer.cur_lr(),
                    iter = self.iter,
                    total_loss = loss.item(),
                    ))
                if jt.rank == 0:
                    self.logger.log(data)           
            self.iter+=1
        
        assert len(files) == len(set(files)),f"{len(files)} {len(set(files))}"
        
        if jt.rank == 0:
            data = metric.value()
            data.update(dict(
                iter = self.epoch,
                name = self.cfg.name,
                is_print = True 
            ))
            self.logger.log(data)


    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        self.model.eval()
        if self.val_dataset is None:
            self.logger.print_log("Please set Val dataset")
            return 
        is_save = True if self.cfg.save_val == True else False
        results = []
        
        metric = IoU(mode="val",ignore_index=self.cfg.ignore_index)
        
        files = []
        for batch_idx,(feats,targets) in tqdm(enumerate(self.val_dataset),desc=f'Val {self.epoch}',total=len(self.val_dataset)):
            outputs = self.model(feats)
            metric.add(outputs,targets)
            files.extend(feats.get("mesh_files"))
            if is_save:
                for mesh_file,o,t in zip(feats.get("mesh_files"),outputs.numpy(),targets.numpy()):
                    results.append((mesh_file,o,t))
        assert len(files) == len(set(files))
        if is_save:
            save_file = osp.join(self.work_dir,"val",f"{self.epoch}.pkl")
            os.makedirs(osp.join(self.work_dir,"val"),exist_ok=True)
            jt.save(results,save_file)
        
        
        data = metric.value()
        data.update(dict(
            iter = self.epoch,
            name = self.cfg.name,
            is_print = True
        ))
        self.logger.log(data)

    @jt.no_grad()
    @jt.single_process_scope()
    def val_iters(self):
        self.model.eval()
        if self.val_dataset is None:
            self.logger.print_log("Please set Val dataset")
            return 
        is_save = True if self.cfg.save_val == True else False
        results = []
        
        if self.cfg.iou_metric:
            metric = IoU(mode="val",ignore_index=self.cfg.ignore_index)
        else:
            metric = Accuracy(mode="val")
        
        reps = 8 if self.cfg.val_iters is None else self.cfg.val_iters

        all_res = dict()
        for i in range(reps):
            if self.cfg.iou_metric:
                metric2 = IoU(mode="val",ignore_index=self.cfg.ignore_index)
            else:
                metric2 = Accuracy(mode="val")
            for batch_idx,(feats,targets) in tqdm(enumerate(self.val_dataset),desc=f'Val {self.epoch}',total=len(self.val_dataset)):
                outputs = self.model(feats)
                metric2.add(outputs,targets)
                for mesh_file,mf,o,t in zip(feats.get("mesh_files"),feats.Fs.numpy(),outputs.numpy(),targets.numpy()):
                    if mesh_file not in all_res:
                        all_res[mesh_file] = [o[:,:mf],t[:mf]] if self.cfg.processor == "segmentation" else [o,t]
                    else:
                        all_res[mesh_file][0] += o[:,:mf] if self.cfg.processor == "segmentation" else o
            print(f"reps {i}:",metric2.value())
        
        results = []
        for mesh_file,(o,t) in all_res.items():
            metric.add(o,t)
            results.append((mesh_file,o,t))
            

        if is_save:
            save_file = osp.join(self.work_dir,"val",f"{self.epoch}.pkl")
            os.makedirs(osp.join(self.work_dir,"val"),exist_ok=True)
            jt.save(results,save_file)

        data = metric.value()
        data.update(dict(
            iter = self.epoch,
            name = self.cfg.name,
            is_print = True
        ))
        print("Final IoU:", data)
        self.logger.log(data)

    @jt.single_process_scope()
    def save(self):
        save_data = {
            "meta":{
                "jmesh_version": jmesh.__version__,
                "epoch": self.epoch,
                "iter": self.iter,
                "max_epoch": self.max_epoch,
                "save_time":current_time(),
                "config": self.cfg.dump()
            },
            "model":self.model.state_dict(),
            "scheduler": self.scheduler.parameters() if self.scheduler else None,
            "optimizer": self.optimizer.parameters()
        }

        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{self.epoch}.pkl")
        jt.save(save_data,save_file)
    
    def load(self, load_path, model_only=False):
        resume_data = jt.load(load_path)

        if (not model_only):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.iter = meta.get("iter",self.iter)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)
            self.scheduler.load_parameters(resume_data.get("scheduler",dict()))
            self.optimizer.load_parameters(resume_data.get("optimizer",dict()))
        if ("model" in resume_data):
            self.model.load_parameters(resume_data["model"])
        elif ("state_dict" in resume_data):
            self.model.load_parameters(resume_data["state_dict"])
        else:
            self.model.load_parameters(resume_data)
        print(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)