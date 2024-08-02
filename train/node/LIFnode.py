import torch 
import torch.nn as nn
from abc import abstractmethod
from typing import Callable, overload
from . import base
import math

class BaseNode(base.MemoryModule):
    def __init__(self,surrogate_function, v_threshold: float = 1., v_reset: float = 0.,
                  detach_reset: bool = False):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.register_memory('v_threshold', v_threshold)
        self.register_memory('v_reset', v_reset)

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):

        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        #软重置仅仅是将膜电位减去阈值的倍数，而硬重置则是将发放脉冲的神经元的膜电位设置为一个特定的重置值self.v_reset，或者保持不变（对于未发放脉冲的神经元）
        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}'

    def forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike
    
    def forward_GradRefine(self, x: torch.Tensor,norm,scale):
        self.neuronal_charge(x)
        spike = self.neuronal_fire_GradRefine(norm,scale)
        self.neuronal_reset(spike)
        return spike

class LIFNode(BaseNode):
    def __init__(self,surrogate_function, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., detach_reset: bool = False):
        assert isinstance(tau, float) and tau > 1.

        super().__init__(surrogate_function, v_threshold, v_reset, detach_reset)
        self.tau = tau
        self.decay_input = decay_input


    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

        else:
            if self.v_reset is None or self.v_reset == 0.:
                if type(self.v) is float:
                    self.v = x
                else:
                    self.v = self.v * (1 - 1. / self.tau) + x
            else:
                self.v = self.v - (self.v - self.v_reset) / self.tau + x

    def forward(self, x: torch.Tensor):
        return super().forward(x)
    
    def forward_GradRefine(self, x: torch.Tensor,norm,scale):
        return super().forward_GradRefine(x,norm,scale)

class MultiStepLIFNode(LIFNode):
    
    all_neurons = []
    
    def __init__(self, surrogate_function, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., detach_reset: bool = False, backend='torch',norm_list=None,norm_func=None,scale_func=None,norm_diff=None,record_norm=None,clamp_func=None,detach_s=None):
        super().__init__(surrogate_function, tau, decay_input, v_threshold, v_reset, detach_reset)
        self.register_memory('v_seq', None)
        self.relu=nn.ReLU()
        self.backend = backend
        # self.tp_v_seq=[]
        self.norm_list=norm_list
        self.norm_func=norm_func
        self.norm_diff=norm_diff
        self.scale_func=scale_func
        self.record_norm=record_norm
        self.clamp_func=clamp_func
        self.record_norm=record_norm
        self.detach_s=detach_s
        if record_norm:
            self.temp_norm_list=[]
            self.grad_l_1_list=[]
            self.grad_l_list=[]
            # self.delta_v=[]
        
        self.norms = [] 
        self.grads_1 = []
        self.grads_before = []
        self.grads_after = []
        
        self.mask = None
        self.rates = None
        MultiStepLIFNode.all_neurons.append(self)
    def neuronal_charge(self, x: torch.Tensor):
        if self.detach_s==None:
            super().neuronal_charge(x)
        else:
            if type(self.v)==torch.Tensor:
                self.v = (self.v*self.detach_s+self.v.detach()*(1-self.detach_s))*0.5+x*0.5
            else:
                self.v = self.v*0.5+x*0.5


    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1 
        if self.record_norm:
            self.grad_l_1=[torch.tensor(1.,requires_grad=True,device=x_seq.device) for i in range(x_seq.shape[0])]
            self.grad_l=[torch.tensor(1.,requires_grad=True,device=x_seq.device) for i in range(x_seq.shape[0])]
            # self.grad_l=[torch.ones((x_seq[0].shape),requires_grad=True,device=x_seq.device) for i in range(x_seq.shape[0])]
            self.grad_l_1_list.append(self.grad_l_1)
            self.grad_l_list.append(self.grad_l)
        if self.norm_list!=None:
            if self.norm_list==-1:
                return self.forward_GradRefine_Standard_Deviation(x_seq)
            else:
                assert x_seq.shape[0]==len(self.norm_list)
                return self.forward_GradRefine(x_seq)
        else:
            return self.forward_origin(x_seq)

    @staticmethod
    def calculate_spike_rates(spike_seq):
        rates = []
        # 获取张量中的元素总数
        total_elements = spike_seq[0].numel()
        for tensor in spike_seq:
            # 计算每个张量中1的个数
            count_ones = tensor.sum().item()
            # 计算脉冲发射率
            spike_rate = count_ones / total_elements
            rates.append(spike_rate)
        return rates
    
    def forward_origin(self, x_seq: torch.Tensor):
        spike_seq = []
        self.v_seq = []
        if self.record_norm:
            temp=self.norm_list
            self.norm_list=-1
        for t in range(x_seq.shape[0]):
            if self.record_norm:
                spike_seq.append(self.forward_GradRefine_Standard_Deviation_step(x_seq[t],self.grad_l_1[t],self.grad_l[t]).unsqueeze(0))
                self.norm_list='-1'
            else:
                spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
        self.v_seq.append(self.v)
        if self.record_norm:
            self.norm_list=temp
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.stack(self.v_seq, 0)
        # self.tp_v_seq.append(self.v_seq)
        # self.rates = self.calculate_spike_rates(spike_seq)
        return spike_seq
        
    def forward_GradRefine(self, x_seq: torch.Tensor):
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            if self.record_norm:
                spike_seq.append(self.forward_GradRefine_Standard_Deviation_step(x_seq[t],self.grad_l_1[t],self.grad_l[t]).unsqueeze(0))
            else:
                spike_seq.append(super().forward_GradRefine(x_seq[t],self.norm_list[t],self.scale_func(self.norm_list[t])).unsqueeze(0))
            self.v_seq.append(self.v)
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.stack(self.v_seq, 0)
        # self.tp_v_seq.append(self.v_seq)
        return spike_seq

    def forward_GradRefine_Standard_Deviation(self, x_seq: torch.Tensor):
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            if self.record_norm:
                spike_seq.append(self.forward_GradRefine_Standard_Deviation_step(x_seq[t],self.grad_l_1[t],self.grad_l[t]).unsqueeze(0))
            else:
                spike_seq.append(self.forward_GradRefine_Standard_Deviation_step_1(x_seq[t], t).unsqueeze(0))
            self.v_seq.append(self.v)
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.stack(self.v_seq, 0)
        self.norm_list=-1
        # self.tp_v_seq.append(self.v_seq)
        # self.rates = self.calculate_spike_rates(spike_seq)
        return spike_seq
    

    def forward_GradRefine_Standard_Deviation_step(self, x: torch.Tensor,grad_l_1=None,grad_l=None):
        self.neuronal_charge(x)
        # self.delta_v.append((self.v.detach() - self.v_threshold).cpu())
        if self.norm_list==-1:
            self.norm_list=self.norm_diff(self.v.detach() - self.v_threshold)
            grad = self.norm_diff(self.ATAN(self.v.detach() - self.v_threshold, 1))
            self.grads_1.append(grad.item())
            norm,scale=1.,1.
            if self.record_norm:
                # self.temp_norm_list.append([float(norm)])
                self.temp_norm_list.append([(float(((self.v.detach() - self.v_threshold)**2).sum()**(1/2)),float((self.v.detach() - self.v_threshold).abs().sum()))])
        elif self.norm_list=='-1':
            norm,scale=1.,1.
            if self.record_norm:
                self.temp_norm_list[-1].append((float(((self.v.detach() - self.v_threshold)**2).sum()**(1/2)),float((self.v.detach() - self.v_threshold).abs().sum())))
        else:
            norm=self.norm_func(self.norm_diff(self.v.detach() - self.v_threshold)/self.norm_list)
            if norm<1: norm = 1
            # self.grads_before.append(self.norm_diff(self.ATAN(self.v.detach() - self.v_threshold, 1)).item())
            # self.grads_after.append(self.norm_diff(self.ATAN(self.v.detach() - self.v_threshold, norm)).item())
            norm=self.clamp_func(norm)
            scale=self.scale_func(norm)
            if self.record_norm:
                self.temp_norm_list[-1].append((float(((self.v.detach() - self.v_threshold)**2).sum()**(1/2)),float((self.v.detach() - self.v_threshold).abs().sum())))
        spike = self.neuronal_fire_GradRefine(norm,scale,grad_l_1,grad_l)
        self.neuronal_reset(spike)
        return spike
    
    @staticmethod
    def ATAN(x, norm, scale=1, alpha=2):
        grad_x = scale * alpha / 2 / (1 + (math.pi / 2 * alpha * x / norm ).pow_(2)) 
        return grad_x
    
    @staticmethod
    def heaviside(x: torch.Tensor):
        return (x >= 0).to(x)
    
    def forward_GradRefine_Standard_Deviation_step_1(self, x: torch.Tensor, t, grad_l_1=None,grad_l=None):
        self.neuronal_charge(x)
        v_adjusted = self.v - self.v_threshold
        v_detached = v_adjusted.detach()  # 只调用一次 detach()
        
        if self.norm_list==-1:
            norm_value = self.norm_diff(v_detached)
            self.norm_list = norm_value.item() / v_detached.numel()
            self.mask = self.heaviside(v_detached)
            
            # 源代码，未优化显存
            # self.norm_list = self.norm_diff(self.v.detach() - self.v_threshold) / (self.v.detach() - self.v_threshold).numel()
            # self.mask = self.heaviside(self.v.detach() - self.v_threshold)
            
            # grad = self.norm_diff(self.ATAN(self.v.detach() - self.v_threshold, 1))
            # self.grads_1.append(grad.item())
            norm,scale=1.,1.

        elif self.norm_list=='-1':
            norm,scale=1.,1.
            if self.record_norm:
                self.temp_norm_list[-1].append((float(((self.v.detach() - self.v_threshold)**2).sum()**(1/2)),float((self.v.detach() - self.v_threshold).abs().sum())))
                
        else:
            mask_inv = 1 - self.mask
            norm = self.norm_diff(v_adjusted * mask_inv) / mask_inv.sum() / self.norm_list
            norm = self.norm_func(norm).item()
            if norm < 1:
                norm = 1
            
            # 源代码，未优化显存
            # norm = ( self.norm_diff((self.v.detach() - self.v_threshold) * (1 - self.mask))/(1 - self.mask).sum() )/ self.norm_list
            # norm=self.norm_func(norm).item()
            # if norm<1: norm = 1
            
            scale = 1
        
        # 记录不同时间步的norm值
        # self.norms.append(norm)
        
        spike = self.neuronal_fire_GradRefine(norm,scale,grad_l_1,grad_l)
        self.mask = self.heaviside(v_detached)  # 更新mask发生在所有计算之后
        self.neuronal_reset(spike)
        
        return spike

    def neuronal_fire_GradRefine(self,norm,scale,grad_l_1=None,grad_l=None):
        # return self.surrogate_function(self.v - self.v_threshold,norm,scale,grad_l_1,grad_l)
        return self.surrogate_function((self.v - self.v_threshold),norm,scale,grad_l_1,grad_l) * (1 - self.mask) + self.surrogate_function((self.v - self.v_threshold),1,scale,grad_l_1,grad_l) * self.mask

    def extra_repr(self):
        return super().extra_repr() + f', backend={self.backend}'

    def reset(self):
        self.norms = []
        self.grads_1 = []
        self.grads_before = []
        self.grads_after = []
        super().reset()
        
    @staticmethod
    def get_all_neurons():
        """返回包含所有neuron实例的列表"""
        return MultiStepLIFNode.all_neurons
    

    
