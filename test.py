from time import time
start_time=time()
# from train.basic_train import init_train
print(time()-start_time)
from train.model.init_model import init_model
from train.model.batchnorm import init_norm
from train.node.init_node import init_node
from train.utils.init_data import init_data
from train.node import surrogate
from train.utils import utils
import torch.nn as nn
from torch.cuda import amp
import os,time,torch
import sys
import argparse
from torch.utils.tensorboard import SummaryWriter
_seed_ = 3407
import random
random.seed(3407)
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(_seed_)
node_type="LIFnode_csr"
data_loader, data_loader_test = init_data("cifar100")(batch_size=128, path='/home/ma-user/work/lw/CIFAR100')
model = init_model("resnet18")(zero_init_residual=True,num_classes=100, pretrained=False, progress=True, \
                T=4, multi_step_neuron=init_node(node_type),norm_layer=init_norm("3d"), v_threshold=1., surrogate_function=surrogate.ATan(), \
                detach_reset=True,multi_out=True,datasets='cifar100')
a=torch.load("/home/ma-user/work/lw/zhanglan/jelly/spiking_resnet/logs/2023-12-16_00:27:16_datasets_cifar100/checkpoint_max_test_acc1.pth")
model.load_state_dict(a['model'])
device="cuda:3"
model.to(device)
from train.node import functional
with torch.no_grad():
    for image, _ in data_loader:
        image= image.to(device)
        model(image.detach())
        functional.reset_net(model)

def find_instances_with_path(module, cls, parent_name=''):
    instances = {}
    for name, child in module.named_children():
        path = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, cls):
            instances[path] = child
        instances.update(find_instances_with_path(child, cls, path))
    return instances
# find_instances_with_path(model, init_node("LIFnode_csr_t"))
y= find_instances_with_path(model, init_node(node_type))
for instance in y:
    data_res18.append(str(instance)+" "+str([float(p) for p in y[instance].spike_rate()]))