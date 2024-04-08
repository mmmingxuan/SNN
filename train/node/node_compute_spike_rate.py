from . import LIFnode, functional
from .LIFAct import *
from typing import Callable, overload
import torch
import torch.nn as nn
class LIFnode_csr(LIFnode.LIFNode):
    def __init__(self, surrogate_function, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., detach_reset: bool = False, backend='torch',ST_target=0, lava_s_cale=1 << 6):
        super().__init__( surrogate_function,tau, decay_input, v_threshold, v_reset, detach_reset)
        self.register_memory('v_seq', None)
        self.ST_target=ST_target
        # check_backend(backend)
        self.relu=nn.ReLU()
        self.backend = backend
        # self.tp_v_seq=[]
        self.lava_s_cale = lava_s_cale

        self.sums=0
        self.spikes=0

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            if self.ST_target==0:
                self.v_seq.append(self.v)
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.stack(self.v_seq, 0)
        self.sums+=self.v_seq.numel()
        self.spikes+=spike_seq.sum()
        # self.tp_v_seq.append(self.v_seq)
        return spike_seq
    
    def spike_rate(self):
        return self.spikes/self.sums

    def spike_rate_reset(self):
        self.sums=0
        self.spikes=0
        

class LIFAct_csr(nn.Module):
    def __init__(self, channel, step=4,**kwargs):
        super(LIFAct_csr, self).__init__()
        self.step = step
        self.V_th = 1.0
        self.temp = 3.0
        self.spikes=0
        self.sums=0
    
    def forward(self, x):
        self.step = x.shape[0]
        u = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            u, out_i = mem_update_withoutbn(x_in=x[i], mem=u, V_th=self.V_th, decay=0.25, temp=self.temp)
            out += [out_i]
        out = torch.stack(out)
        self.spikes+=out.sum()
        self.sums+=out.numel()
        return out

    def spike_rate(self):
        return self.spikes/self.sums

    def spike_rate_reset(self):
        self.sums=0
        self.spikes=0

class LIFnode_csr_t(LIFnode.LIFNode):
    def __init__(self, surrogate_function, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0.,
                 detach_reset: bool = False, backend='torch',ST_target=0, lava_s_cale=1 << 6):
        super().__init__(surrogate_function,tau, decay_input, v_threshold, v_reset,  detach_reset)
        self.register_memory('v_seq', None)
        self.ST_target=ST_target
        # check_backend(backend)
        self.relu=nn.ReLU()
        self.backend = backend
        # self.tp_v_seq=[]
        self.lava_s_cale = lava_s_cale
        self.sums=[0]
        self.spikes=[0]

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        
        spike_seq = []
        self.v_seq = []
        if len(self.spikes)==1:
            self.spikes*=x_seq.shape[0]
            self.sums*=x_seq.shape[0]
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            if self.ST_target==0:
                self.v_seq.append(self.v)
            self.sums[t]+=spike_seq[-1].numel()
            self.spikes[t]+=spike_seq[-1].sum()
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.stack(self.v_seq, 0)
        # self.tp_v_seq.append(self.v_seq)
        return spike_seq
    
    def spike_rate(self):
        return [spikess/sumss for spikess,sumss in zip(self.spikes,self.sums)]

    def spike_rate_reset(self):
        self.sums=[0]
        self.spikes=[0]
        

class LIFAct_csr_t(nn.Module):
    def __init__(self, channel, step=4,**kwargs):
        super(LIFAct_csr_t, self).__init__()
        self.step = step
        self.V_th = 1.0
        self.temp = 3.0
        self.spikes=[0]
        self.sums=[0]
    
    def forward(self, x):
        self.step = x.shape[0]
        if len(self.spikes)==1:
            self.spikes*=self.step
            self.sums*=self.step
        u = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            u, out_i = mem_update_withoutbn(x_in=x[i], mem=u, V_th=self.V_th, decay=0.25, temp=self.temp)
            out += [out_i]
            self.sums[i]+=out_i.numel()
            self.spikes[i]+=out_i.sum()
        out = torch.stack(out)
        return out

    def spike_rate(self):
        return [spikess/sumss for spikess,sumss in zip(self.spikes,self.sums)]

    def spike_rate_reset(self):
        self.sums=[0]
        self.spikes=[0]