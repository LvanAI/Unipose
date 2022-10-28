import mindspore
import mindspore.nn as nn


from mindspore import load_checkpoint, load_param_into_net
from mindspore.common.initializer import HeNormal

from .wasp import build_wasp
from .decoder import build_decoder
from .resnet import ResNet50


def build_backbone(backbone, output_stride):
    if backbone == 'resnet':
        return ResNet50(output_stride)
    else:
        raise NotImplementedError

class unipose(nn.Cell):
    def __init__(self, args):
                   
        super(unipose, self).__init__()
        
        self.stride = args.stride
        self.heatmap_h = args.heatmap_h
        self.heatmap_w = args.heatmap_w
 

        self.num_classes = args.num_classes

        self.backbone = build_backbone(args.backbone, args.output_stride)
        if args.pretrained:
            print('load pretrained weigths from %s'%(args.pretrained))
            param_dict = load_checkpoint(args.pretrained)
            param_dicts = {}
            for key, value in param_dict.copy().items():
                current_key = "backbone." + key.replace("down_sample_layer", "downsample")
                param_dicts[current_key] = value
            load_param_into_net(self.backbone, param_dicts)

        self.wasp = build_wasp(args.output_stride)
        self.decoder  = build_decoder(args.num_classes, args.backbone)


    def construct(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)

        if x.shape[0] != self.heatmap_h or x.shape[1] != self.heatmap_w:
             x = nn.ResizeBilinear()(x, size=(self.heatmap_h,self.heatmap_w), align_corners=True)
        return x





