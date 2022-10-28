import os

import math
import mindspore
import mindspore.nn as nn
import mindspore.context as context
# import mindspore.nn.functional as F
import mindspore.ops as ops
import mindspore.common.initializer as initializer
from mindspore.common.initializer import HeNormal

import mindspore.numpy as np
from mindspore import dtype as mstype


# if os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
#     BatchNorm2d = nn.SyncBatchNorm
# else:
#     BatchNorm2d = nn.BatchNorm2d

BatchNorm2d = nn.BatchNorm2d

class AdaptiveAvgPool2D(nn.Cell):
    def __init__(self):
        super(AdaptiveAvgPool2D, self).__init__()

    def construct(self, input):
        H = input.shape[-2]
        W = input.shape[-1]
        x = ops.AvgPool(kernel_size=(H,W))(input)
        return x

class _AtrousModule(nn.Cell):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_AtrousModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, pad_mode='pad', weight_init=HeNormal(),
                                     stride=1, padding=padding, dilation=dilation, has_bias=False)
        self.bn = BatchNorm2d(planes,)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

 

class wasp(nn.Cell):
    def __init__(self, output_stride):
        super(wasp, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [48, 24, 12, 6]

        elif output_stride == 8:
            dilations = [48, 36, 24, 12]
        else:
            raise NotImplementedError

        self.aspp1 = _AtrousModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _AtrousModule(256, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _AtrousModule(256, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _AtrousModule(256, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.SequentialCell(AdaptiveAvgPool2D(),
                                                 nn.Conv2d(inplanes, 256, 1, stride=1, has_bias=False, pad_mode='pad',weight_init=HeNormal()),          
                                                 BatchNorm2d(256),
                                                 nn.ReLU())


        self.conv1 = nn.Conv2d(1280, 256, 1, has_bias=False, pad_mode='pad', weight_init=HeNormal())
        self.conv2 = nn.Conv2d(256, 256, 1, has_bias=False, pad_mode='pad', weight_init=HeNormal())
        self.bn1 = BatchNorm2d(256, )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def construct(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x5 = self.global_avg_pool(x)
        x5 = nn.ResizeBilinear()(x5, size=x4.shape[2:], align_corners=True)
        x = ops.Concat(axis=1)((x1, x2, x3, x4, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

 


def build_wasp(output_stride):
    return wasp(output_stride)



