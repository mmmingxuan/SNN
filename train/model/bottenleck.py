import torch.nn as nn
from ..node import functional
from ..node.LIFAct import LIFAct, LIFAct_withoutbn
from .batchnorm import myBatchNorm3d, myBatchNorm2d
from .conv import conv1x1, conv3x3


class bottleneck(nn.Module):
    def __init__(
        self, channel_in, channel_out, kernel_size, node, norm, groups=1, **kwargs
    ):
        super(bottleneck, self).__init__()
        middle_channel = channel_out
        self.cv1 = nn.Conv2d(
            channel_in,
            middle_channel,
            kernel_size=kernel_size,
            stride=kernel_size,
            groups=groups,
        )
        self.bn1 = norm(middle_channel)
        if "LIFAct" in str(node):
            self.nv1 = node(middle_channel)
        else:
            self.nv1 = node(**kwargs)

        self.cv2 = conv3x3(middle_channel, middle_channel, groups=groups)
        self.bn2 = norm(middle_channel)
        if "LIFAct" in str(node):
            self.nv2 = node(middle_channel)
        else:
            self.nv2 = node(**kwargs)

        self.cv3 = conv3x3(middle_channel, channel_out, groups=groups)
        self.bn3 = norm(channel_out)
        if "LIFAct" in str(node):
            self.nv3 = node(channel_out)
        else:
            self.nv3 = node(**kwargs)

    def forward(self, x_seq):
        x_seq = functional.seq_to_ann_forward(x_seq, [self.cv1])
        x_seq = self.bn1(x_seq)
        x_seq = self.nv1(x_seq)

        identity = x_seq

        out = functional.seq_to_ann_forward(x_seq, [self.cv2])
        x_seq = self.bn2(x_seq)
        out = self.nv2(out)

        out = functional.seq_to_ann_forward(out, [self.cv3])
        x_seq = self.bn3(x_seq)
        out = self.nv3(out + identity)

        return out
