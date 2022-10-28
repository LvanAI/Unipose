import os
import math
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from mindspore.common.initializer import initializer, Normal, HeNormal
from mindspore import load_checkpoint, load_param_into_net


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

# if os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
#     BatchNorm2d = nn.SyncBatchNorm
# else:
#     BatchNorm2d = nn.BatchNorm2d

BatchNorm2d = nn.BatchNorm2d

class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False,pad_mode='pad',
                               weight_init=Normal(sigma=math.sqrt(2. / planes), mean=0.0))
        self.bn1 = BatchNorm2d(planes, )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, has_bias=False,pad_mode='pad',
                               weight_init=Normal(sigma=math.sqrt(2. / (planes * 9)), mean=0.0))
        self.bn2 = BatchNorm2d(planes,)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False,pad_mode='pad',
                               weight_init=Normal(sigma=math.sqrt(2. / (planes * 4) ), mean=0.0))
        self.bn3 = BatchNorm2d(planes * 4,)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Cell):

    def __init__(self, block, layers, output_stride):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            #strides = [1, 1, 1, 1]
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,pad_mode='pad',
                                has_bias=False,
                               weight_init=Normal(sigma=math.sqrt(2. / (64 * 49)), mean=0.0))
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,pad_mode='pad',
                          kernel_size=1, stride=stride, has_bias=False,
                               weight_init=Normal(sigma=math.sqrt(2. / (planes  * block.expansion)), mean=0.0)),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.SequentialCell(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,pad_mode='pad',
                          kernel_size=1, stride=stride, has_bias=False,
                               weight_init=Normal(sigma=math.sqrt(2. / (planes  * block.expansion)), mean=0.0)),
                BatchNorm2d(planes * block.expansion, ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation))

        return nn.SequentialCell(*layers)

    def construct(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat



def ResNet101(output_stride):
    """Constructs a ResNet-101 model.
    Args:
    """

    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride)


    return model

def ResNet50(output_stride):
    """Constructs a ResNet-50 model
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride)
    return model

if __name__ == "__main__":


    net = ResNet101( output_stride=8)

    ckpt_file_name = "adjust_param_dict.ckpt"
    param_dict = load_checkpoint(ckpt_file_name)
    load_param_into_net(net, param_dict)

    input = mindspore.numpy.rand(1, 3, 512, 512)
    output, low_level_feat = net(input)
    print('RESNET output.shape() ： ',output.shape) # (1, 2048, 64, 64)
    print('RESNET low_level_feat.shape() ： ',low_level_feat.shape) # (1, 256, 127, 127)
    print("RESNET OK!")