import os 
# fix trimesh slow multi thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import jittor as jt
from jmesh.runner import Runner 
from jmesh.config import init_cfg

def main():
    parser = argparse.ArgumentParser(description="Jittor Mesh  Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="train",
        help="train,val",
        type=str,
    )

    parser.add_argument(
        "--no_cuda",
        action='store_true'
    )
    
    args = parser.parse_args()

    if not args.no_cuda:
        jt.flags.use_cuda=1
        jt.cudnn.set_max_workspace_ratio(0.0)

    assert args.task in ["train","val","val_iters"],f"{args.task} not support, please choose [train,val,val_iters]"
    
    if args.config_file:
        init_cfg(args.config_file)

    runner = Runner()

    if args.task == "train":
        runner.run()
    elif args.task == "val":
        runner.val()
    elif args.task == "val_iters":
        runner.val_iters()

if __name__ == "__main__":
    main()