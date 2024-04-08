from .data_loader import *
data_dict={
    'cifar100':get_cifar100_data,
    'cifar100_aug':get_cifar100_data_aug,
    'cifar10_dvs':get_cifar10dvs_data,
}

def init_data(data):
    return data_dict[data]