# Accurate Instance Segmentation for Remote Sensing Images via Adaptive and Dynamic Feature Learning

by Feng Yang, Xiangyue Yuan, Jie Ran, Wenqiang Shu, Yue Zhao, Anyong Qin and Chenqiang Gao.

### Introduction

This repository is for the paper "Accurate Instance Segmentation for Remote Sensing Images via Adaptive and Dynamic Feature Learning" whhich is constructed based on [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).


### Installation

For environment requirements, data preparation and compilation, please refer to [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).

WARNING: pytorch 0.4.1 is broken, see https://github.com/pytorch/pytorch/issues/8483. Use pytorch 0.4.0

### Usage

For training and testing, we keep the same as the one in [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).

```shell
python tools/train_net_step.py --dataset coco2017 --cfg configs/rs/e2e_rs_R-50-FPN_2x_mask.yaml
```

To evaluate the model, simply use:

```shell
python tools/test_net.py --dataset coco2017 --cfg configs/rs/e2e_rs_R-50-FPN_2x_mask.yaml --load_ckpt {path/to/your/checkpoint}
```


### Questions

Please contact 's200101052@stu.cqupt.edu.cn'
