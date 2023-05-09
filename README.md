# Mesh Neural Networks Based on Dual Graph Pyramids(TVCG)

The repository contains official jittor implementations for **LRPNet**. 

The paper is in [Here](https://arxiv.org/pdf/2301.06962).


## Installation
```
pip install -r requirements.txt
```

## Scannet
Download the scannet and prepare it.

Change tools/process_scannet.py "data_dir"
```bash
bash run_prepare.sh
```

## Models
We release our trained models and training logs in "work_dirs".

## Evaluation
To evaluate the model, run:

```bash
bash run.sh 0 configs/tvcg_scene_scannetv2_val.py --task=val_iters
# for test 
bash run.sh 0 configs/tvcg_scene_scannetv2_test.py --task=val_iters
```
For final prediction:
```bash
# change the dir before merging.
python3 tools/merge_results.py
```
## Train 

```bash
bash dist_run.sh 0,1,2,3 4 configs/tvcg_scene_scannetv2_train.py --task=train
```

## Citation
If you find our repo useful for your research, please consider citing our paper:

```
@article{li2023mesh,
  title={Mesh neural networks based on dual graph pyramids},
  author={Li, Xiang-Li and Liu, Zheng-Ning and Chen, Tuo and Mu, Tai-Jiang and Martin, Ralph R and Hu, Shi-Min},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2023},
  publisher={IEEE}
}
```

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
