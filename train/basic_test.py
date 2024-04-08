from torch.cuda import amp
import torch
from .utils import utils
import time
from .node import functional
import math
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
class model_test():
    def __init__(self,model=None,device="cuda:1",data_loader=None,data_loader_test=None,loss_func=None):
        super(model_test,self).__init__()
        self.model=model
        self.device=device
        self.data_loader=data_loader
        self.data_loader_test=data_loader_test
        self.loss_func=loss_func
        self.model.eval()
        if type(self.device)==list:
            self.distributed=True
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=self.device)
            self.model_without_ddp = self.model.module
        else:
            self.distributed=False
            self.model.to(self.device)
            
    def acc_each_layer(self):
        self.sums=0
        self.acc1_train=[0]
        with torch.no_grad():
            for image, target in self.data_loader:
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(image.detach())
                output = output.mean(dim=1)
                predict = output.argmax(dim=-1)
                self.sums+=int(len(predict[0]))
                if len(self.acc1_train)==1:
                    self.acc1_train*=len(predict)
                for i in range(len(predict)):
                    self.acc1_train[i]+=sum(predict[i]==target)
                functional.reset_net(self.model)

        self.sums_t=0
        self.acc1_test=[0]
        with torch.no_grad():
            for image, target in self.data_loader_test:
                image = image.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(image.detach())
                output = output.mean(dim=1)
                predict = output.argmax(dim=-1)
                self.sums_t+=int(len(predict[0]))
                if len(self.acc1_test)==1:
                    self.acc1_test*=len(predict)
                for i in range(len(predict)):
                    self.acc1_test[i]+=sum(predict[i]==target)
                functional.reset_net(self.model)

        return [int(p)/self.sums for p in self.acc1_train],[int(p)/self.sums_t for p in self.acc1_test]
    
    def spike_rate(self):
        self.sr=[]
        def find_instances_with_path(module, cls, parent_name=''):
            instances = {}
            for name, child in module.named_children():
                path = f"{parent_name}.{name}" if parent_name else name
                if isinstance(child, cls):
                    instances[path] = child
                instances.update(find_instances_with_path(child, cls, path))
            return instances
            
        y= find_instances_with_path(model, init_node("LIFnode_csr"))
        for instance in y:
            self.sr.append(str(instance)+" "+str([float(p) for p in y[instance].spike_rate()]))
        return self.sr
    
    def plot_spike_rate(self,title):
        self.spike_rate()
        names = []
        values = []
        for line in self.sr:
            parts = line.split(' ', 1)
            name = parts[0]
            vals = eval(parts[1])
            names.append(name)
            values.append(vals)

        # 计算每组柱状图的位置
        n = len(values[0])  # 每个名称对应的数值个数
        x = np.arange(len(names))  # 名称数量
        width = 0.2  # 柱状图的宽度

        # 创建一个图表
        plt.figure(figsize=(15, 10))

        # 绘制柱状图
        for i in range(n):
            plt.bar(x - width/2. + i*width, [val[i] for val in values], width, label=f'T= {i+1}')

        # 设置图表的标题和标签
        # plt.title('Resnet18 T=4 LIFnode Train spike rate')
        plt.title(title)
        plt.ylabel('Value')
        plt.xticks(x, names, rotation=45)
        plt.legend()

        # 显示图表
        plt.tight_layout()
        plt.show()



        



