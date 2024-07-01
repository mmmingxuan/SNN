import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

class SurrogateFunctionBase(nn.Module):
    def __init__(self, alpha, spiking=True):
        super().__init__()
        self.spiking = spiking
        self.alpha = alpha

    def set_spiking_mode(self, spiking: bool):
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    @staticmethod
    def spiking_function(x, alpha):
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha):
        raise NotImplementedError

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        raise NotImplementedError

    def cuda_code_start_comments(self):
        return f'// start: spikingjelly.clock_driven.surrogate.{self._get_name()}.cuda_code'

    def cuda_code_end_comments(self):
        return f'// end: spikingjelly.clock_driven.surrogate.{self._get_name()}.cuda_code'

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]).pow_(2)) * grad_output

        return grad_x, None

class ATan(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)


    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5
    
class atan_GradRefine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, norm_factor, scale_factor):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.norm_factor = norm_factor
            ctx.scale_factor = scale_factor
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]: 
            grad_x = ctx.scale_factor * ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]/ ctx.norm_factor ).pow_(2)) * grad_output
        
        return grad_x, None,None,None
                                                                                                  
class atan_GradRefine_rgrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, norm_factor, scale_factor,grad_l_1=None, grad_l=None):
        # print(grad_l_1.device,grad_l.device)
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            ctx.norm_factor = norm_factor
            ctx.scale_factor = scale_factor
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]: 
            grad_self=ctx.scale_factor * ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * ctx.saved_tensors[0]/ ctx.norm_factor ).pow_(2)) 
            grad_x =grad_self * grad_output

        return grad_x, None,None,None,torch.norm(grad_x.clone().to(torch.float32),p=2),torch.norm(grad_self.clone().to(torch.float32),p=2)#,grad_self.to(torch.float32)

class ATan_GradRefine(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)


    @staticmethod
    def spiking_function(x, alpha, norm_factor=1., scale_factor=1.,grad_l_1=None,grad_l=None):
        if grad_l_1==None and grad_l==None:
            return atan_GradRefine.apply(x, alpha, norm_factor, scale_factor)
        else:
            return atan_GradRefine_rgrad.apply(x, alpha, norm_factor, scale_factor,grad_l_1,grad_l)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (math.pi / 2 * alpha * x).atan_() / math.pi + 0.5

    def forward(self, x: torch.Tensor,norm,scale,grad_l_1=None,grad_l=None):
        if self.spiking:
            return self.spiking_function(x, self.alpha,norm,scale,grad_l_1,grad_l)

        else:
            return self.primitive_function(x, self.alpha,norm,scale)