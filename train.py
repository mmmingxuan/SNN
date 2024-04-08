from train.basic_train import init_train
from train.model.init_model import init_model
from train.model.batchnorm import init_norm
from train.node.init_node import init_node
from train.node.norm_scale_func import *
from train.utils.init_data import init_data
from train.node import atan
from train.node import LIFnode
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
import wandb



def main(args):

    #surrogate
    norm_func=default_norm_func
    scale_func=default_scale_func
    clamp_func=clamp_func_default
    norm_diff=default_norm_diff
    if args.norm_list!=None:
        surrogate_function=atan.ATan_GradRefine()
    else:
        surrogate_function=atan.ATan() 
    if args.norm_list!=None:
        if args.norm_list[-1]=="+":
            assert len(args.norm_list)==3
            args.norm_list=arithmetic_series(float(args.norm_list[0]),float(args.norm_list[1]),args.T)
        elif args.norm_list[-1]=="*":
            assert len(args.norm_list)==3
            args.norm_list=geometric_series(float(args.norm_list[0]),float(args.norm_list[1]),args.T)
        elif args.norm_list[-1]=="std1":
            args.norm_list=-1
        elif args.norm_list[-1]=="std2":
            args.norm_list=-1
            norm_func=lambda norm:reciprocal_norm_func(norm,alpha=args.reci_w)
        elif args.norm_list[-1]=="std3":
            args.norm_list=-1
            norm_func=lambda norm:linear_norm_func(norm,reci_w=args.reci_w)
        elif args.norm_list[-1]=="std4":
            args.norm_list=-1
            norm_func=lambda norm:ln_norm_func(norm,reci_w=args.reci_w)
        else:
            args.norm_list=[float(norm) for norm in args.norm_list]

    if args.scale_generate == "1":
        scale_func=reciprocal_scale_func
    elif args.scale_generate == "2":
        scale_func=equal_scale_func

    if args.norm_diff == "1":
        norm_diff=double_norm_diff
    elif args.norm_diff == "2":
        norm_diff=abs_norm_diff
    
    if args.clamp_func!=None and args.clamp_func[0]=="max":
        clamp_func=lambda norm:clamp_func_max(norm,alpha=float(args.clamp_func[1]))
    elif args.clamp_func!=None and args.clamp_func[0]=="min":
        clamp_func=lambda norm:clamp_func_min(norm,alpha=float(args.clamp_func[1]))
    
    #output
    output_dir = os.path.join("./logs", f'{time.strftime("%Y-%m-%d_%H:%M:%S")}_datasets_{args.dataset}')
    utils.mkdir(output_dir)
    if args.redirect:
        outputs_dir = os.path.join("./log", f'{time.strftime("%Y-%m-%d_%H:%M:%S")}_datasets_{args.dataset}.txt')
        print(f"all output have been tranferred into {outputs_dir}")
        sys.stdout=open(outputs_dir, 'w', buffering=1)

    #dataset
    if "cifar10_dvs" in args.dataset:
        args.T=10
    data_loader, data_loader_test = init_data(args.dataset+("_aug" if args.aug else ""))(batch_size=args.batch_size, path=args.data_path,T=args.T,data_path=args.data_path)
    
    #model
    if args.model=="braincog":
        from timm.models import create_model
        model = create_model(model_name='resnet18',
            num_classes=10,
            adaptive_node=False,
            dataset='dvsc10',
            step=10,
            encode_type='direct',
            node_type=LIFnode.LIFNode,
            threshold=0.5,
            tau=2.0,
            sigmoid_thres=False,
            requires_thres_grad=False,
            spike_output=False,
            act_fun='AtanGrad',
            temporal_flatten=False,
            layer_by_layer=False,
            n_groups=1,
            n_encode_type='linear',
            n_preact=False,
            tet_loss=False,
            sew_cnf='ADD',
            conv_type='normal',
            )
    else:
        model = init_model(args.model)(zero_init_residual=True,num_classes=args.num_classes, pretrained=False, progress=True, \
                T=args.T, multi_step_neuron=init_node(args.node),norm_layer=init_norm(args.norm_layer), v_threshold=args.v_threshold, surrogate_function=surrogate_function, \
                detach_reset=True,multi_out=True,datasets=args.dataset,norm_list=args.norm_list,norm_func=norm_func,scale_func=scale_func,norm_diff=norm_diff,\
                record_norm=args.record_norm,clamp_func=clamp_func,detach_s=args.detach_s)

    print(model)
    wandb.init(project="SNN",name=args.loss_type)
    wandb.watch(model, log="all")
    #amp
    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    #optimizer
    if args.optimizer=="adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.optimizer=="adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.optimizer=="braincog":
        from timm.optim import create_optimizer_v2
        from timm.scheduler import create_scheduler
        args.lr=0.0003125
        optimizer=create_optimizer_v2(model,'adamw',0.000625,1e-4,0.9,eps=1e-8)
        lr_scheduler, _ = create_scheduler(args, optimizer)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(optimizer)

    #start log
    purge_step_train = 0
    purge_step_te = 0
    train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
    te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
    with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
    print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    #loss_func
    loss_func = nn.CrossEntropyLoss()
    # from timm.loss import LabelSmoothingCrossEntropy
    # loss_func=LabelSmoothingCrossEntropy(smoothing=0.1).cuda()
    
    #define train class
    Train_Loss=init_train(args.loss_type)(model,args.device,lr_scheduler,optimizer,data_loader, data_loader_test,loss_func,args=args,scaler=scaler,train_tb_writer=train_tb_writer,te_tb_writer=te_tb_writer,output_dir=output_dir)
    
    if args.resume:
        Train_Loss.resume(args.resume)
    #train start
    Train_Loss.train()
    # Train_Loss.train_save_diffsum(div=args.div)
    wandb.finish()

    #close redirect
    if args.redirect:
        sys.stdout.close()


def parse_float_list(string):
    return [float(item) for item in string.split(',')]
    
    
def parse_norm_list(string):
    return [item for item in string.split(',')]

def arithmetic_series(min, max, num_elements):
    if num_elements == 1:
        return [min]
    step = (max - min) / (num_elements - 1)
    return [min + step * i for i in range(num_elements)]

def geometric_series(min, max, num_elements):
    if num_elements == 1:
        return [min]
    ratio = (max / min) ** (1 / (num_elements - 1))
    return [min * ratio ** i for i in range(num_elements)]

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    #dataset settings
    parser.add_argument('--dataset', default='cifar100', help='dataset', type=str)
    parser.add_argument('--data_path', default='/home/ma-user/work/lw/CIFAR100', help='dataset', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--aug', action='store_true')
    
    #model settings
    parser.add_argument('--model', default='sew_resnet18', type=str)
    parser.add_argument('--node', default='LIFnode', type=str)
    parser.add_argument('--v_threshold', default=1., type=float)
    parser.add_argument('--norm_layer', default='2d', type=str)

    #surrogate setting
    parser.add_argument('--norm_list', default=None, type=parse_norm_list)
    parser.add_argument('--scale_generate',default=None)
    parser.add_argument('--norm_diff',default=None)
    parser.add_argument('--clamp_func',default=None, type=parse_norm_list)
    parser.add_argument('--reci_w',default=15,type=int)
    parser.add_argument('--record_norm', action='store_true')

    #optimizer settings
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr','--learning_rate', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--sched', default='cosine', type=str)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--warmup_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--cooldown_epochs', default=10, type=int)

    #train settings
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--T', default=4, type=int)
    parser.add_argument('--redirect', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--loss_type', type=str, required=True)
    parser.add_argument('--div', default=1, type=int)
    parser.add_argument('--detach_s', default=None, type=float)

    #resume
    parser.add_argument('--resume', default='', type=str)

    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

