import torch.nn as nn
from ..node import functional

class myBatchNorm3d(nn.Module):
    def __init__(self, num_features, step=4):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features)
        self.step = step
    def forward(self, x):
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()
        return out

class myBatchNorm2d(nn.Module):
    def __init__(self, num_features, step=4):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.step = step
    def forward(self, x):
        out = functional.seq_to_ann_forward(x, self.bn)
        return out

def init_norm(norm_layer):
    return myBatchNorm3d if norm_layer=="3d" else myBatchNorm2d
