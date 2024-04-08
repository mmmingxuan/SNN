import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from IPython import embed
def spike_activation(x, ste=False, temp=1.0):
    out_s = torch.gt(x, 0.5)
    if ste:
        out_bp = torch.clamp(x, 0, 1)
    else:
        out_bp = torch.clamp(x, 0, 1)
        out_bp = (torch.tanh(temp * (out_bp-0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
    return (out_s.float() - out_bp).detach() + out_bp


def mem_update(bn, x_in, mem, V_th, decay, temp=1.0):
    mem = mem * decay + x_in
    
    mem2 = bn(mem)
    spike = spike_activation(mem2/V_th, temp=temp)
    mem = mem * (1 - spike)
    return mem, spike


class LIFAct(nn.Module):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, channel, step=4,**kwargs):
        super(LIFAct, self).__init__()
        self.step = step
        self.V_th = 1.0
        self.temp = 3.0
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        self.step = x.shape[0]
        u = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            u, out_i = mem_update(bn=self.bn, x_in=x[i], mem=u, V_th=self.V_th, decay=0.25, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out

def mem_update_withoutbn(x_in, mem, V_th, decay, temp=1.0):
    mem = mem * decay + x_in
    
    mem2 = mem
    spike = spike_activation(mem2/V_th, temp=temp)
    mem = mem * (1 - spike)
    return mem, spike


class LIFAct_withoutbn(nn.Module):
    """ Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """

    def __init__(self, channel, step=4,**kwargs):
        super(LIFAct_withoutbn, self).__init__()
        self.step = step
        self.V_th = 1.0
        self.temp = 3.0

    def forward(self, x):
        self.step = x.shape[0]
        u = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            u, out_i = mem_update_withoutbn(x_in=x[i], mem=u, V_th=self.V_th, decay=0.25, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        return out
