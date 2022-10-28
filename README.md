# MindSpore Unipose

## Introduction

This work is used for reproduce Unipose based on NPU(Ascend 910)

**Unipose** is introduced in [arxiv](https://arxiv.org/pdf/2001.08095v1.pdf)

## Training


```
mpirun -n 8 python train.py --config <config path> > train.log 2>&1 &
```


## Acknowledgement

We heavily borrow the code from [UniPose](https://github.com/bmartacho/UniPose) and [swin_transformer](https://gitee.com/mindspore/models/tree/master/research/cv/swin_transformer)
We thank the authors for the nicely organized code!
