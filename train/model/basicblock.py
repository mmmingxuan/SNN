from .conv import conv3x3,conv1x1
from .batchnorm import myBatchNorm3d,myBatchNorm2d
from ..node import functional
from ..node.LIFAct import LIFAct,LIFAct_withoutbn
import torch.nn as nn
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, single_step_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if 'LIFAct' in str(single_step_neuron):
            self.sn1=single_step_neuron(planes)
        else:
            self.sn1 = single_step_neuron(**kwargs)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        if 'LIFAct' in str(single_step_neuron):
            self.sn2 = single_step_neuron(planes)
        else:
            self.sn2 = single_step_neuron(**kwargs)
        self.stride = stride
        self.config = kwargs
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.sn2(identity+out)

        return out


class MultiStepBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=myBatchNorm2d, multi_step_neuron: callable = None, **kwargs):
        super().__init__(inplanes, planes, stride, downsample, groups,
                 base_width, dilation, norm_layer, multi_step_neuron, **kwargs)
    def forward(self, x_seq):
        
        identity = x_seq
        out = functional.seq_to_ann_forward(x_seq, self.conv1)
        out = self.bn1(out)
        out = self.sn1(out)

        out = functional.seq_to_ann_forward(out, self.conv2)
        out=self.bn2(out)
            
        if self.downsample is not None:
            identity = self.downsample[1](functional.seq_to_ann_forward(x_seq, self.downsample[0]))

        out = self.sn2(identity+out)
        return out