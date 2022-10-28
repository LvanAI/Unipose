import os
import math
import mindspore
import mindspore.nn as nn
import mindspore.context as context
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal

# if os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
#     BatchNorm2d = nn.SyncBatchNorm
# else:
#     BatchNorm2d = nn.BatchNorm2d
BatchNorm2d = nn.BatchNorm2d

class MaxPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.max_pool = ops.MaxPool(kernel_size, stride)
        self.use_pad = padding != 0
        if isinstance(padding, tuple):
            assert len(padding) == 2
            paddings = ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1]))
        elif isinstance(padding, int):
            paddings = ((0, 0),) * 2 + ((padding, padding),) * 2
        else:
            raise ValueError('padding should be a tuple include 2 numbers or a int number')
        self.pad = ops.Pad(paddings)

    def construct(self, x):
        if self.use_pad:
            x = self.pad(x)
        return self.max_pool(x)

class Decoder(nn.Cell):
    def __init__(self, num_classes, backbone):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, has_bias=False,
                                                 pad_mode='pad',weight_init=HeNormal())
        self.bn1 = BatchNorm2d(48,)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, 256, 1, has_bias=False,
                                                 pad_mode='pad',weight_init=HeNormal())
        self.bn2 = BatchNorm2d(256, )
        self.last_conv = nn.SequentialCell(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, has_bias=False,
                                                 pad_mode='pad',weight_init=HeNormal()),
                                       BatchNorm2d(256,),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, has_bias=False,
                                                 pad_mode='pad',weight_init=HeNormal()),
                                       BatchNorm2d(256,),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes+1, kernel_size=1, stride=1,
                                                 pad_mode='pad',weight_init=HeNormal()))

        self.maxpool = MaxPool2d(kernel_size=3, stride=2,padding=1)


    def construct(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        low_level_feat = self.maxpool(low_level_feat)

        x = nn.ResizeBilinear()(x, size=low_level_feat.shape[2:], align_corners=True)

        x = mindspore.ops.Concat(axis=1)((x, low_level_feat))
        x = self.last_conv(x)
        return x


def build_decoder( num_classes, backbone):
    return Decoder(num_classes, backbone)


