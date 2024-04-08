import math
def default_norm_func(standard_deviation_ratio):
    return standard_deviation_ratio

def reciprocal_norm_func(norm,alpha=15):
    return 1 + (max(0,math.log(norm+1e-10))**2)/alpha

def linear_norm_func(norm,reci_w=2):
    return 1 + (norm-1)/reci_w

###########################################################
def ln_norm_func(norm,reci_w=2):
    return 1 + math.log(norm)/reci_w

def default_scale_func(norm):
    return 1.

def reciprocal_scale_func(norm):
    return 1./norm

def equal_scale_func(norm):
    return norm

###########################################################
def default_norm_diff(delta_v):
    return delta_v.std()

def double_norm_diff(delta_v):
    return (delta_v**2).sum()**(1/2)

def abs_norm_diff(delta_v):
    return delta_v.abs().sum()
###########################################################
def clamp_func_default(norm):
    return norm

def clamp_func_max(norm,alpha=2):
    return alpha if norm >alpha else norm

def clamp_func_min(norm,alpha=1):
    return alpha if norm <alpha else norm