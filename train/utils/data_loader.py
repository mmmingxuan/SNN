import torchvision.datasets as datasets
from timm.data import ImageDataset, create_loader, Mixup, FastCollateMixup, AugMixDataset
from timm.data import create_transform
import torch
import math
import os
import numpy as np
from .autoaugment import *
from .cifar10_dvs import CIFAR10DVS
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
from . import tonic
from .tonic import DiskCachedDataset
DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.2023, 0.1994, 0.2010)


def build_transform(is_train, img_size):
    """
    构建数据增强, 适用于static data
    :param is_train: 是否训练集
    :param img_size: 输出的图像尺寸
    :return: 数据增强策略
    """
    resize_im = img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                img_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * img_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(img_size))

    t.append(transforms.ToTensor())
    if img_size > 32:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, img_size, dataset, same_da=False, path=None):
    """
    构建带有增强策略的数据集
    :param is_train: 是否训练集
    :param img_size: 输出图像尺寸
    :param dataset: 数据集名称
    :param path: 数据集路径
    :param same_da: 为训练集使用测试集的增广方法
    :return: 增强后的数据集
    """
    transform = build_transform(False, img_size) if same_da else build_transform(is_train, img_size)

    if dataset == 'CIFAR100':
        if path is None:
            path = dataset
        print('Get Cifar from ', path)
        dataset = datasets.CIFAR100(path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    else:
        raise NotImplementedError

    return dataset, nb_classes

def get_cifar100_data(batch_size=128, num_workers=8, path=None,**kwargs):
    """
    获取CIFAR100数据
    https://www.cs.toronto.edu/~kriz/cifar.html
    :param batch_size: batch size
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    train_datasets, _ = build_dataset(True, 32, 'CIFAR100', False, path)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR100', False, path)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )
    return train_loader, test_loader

def get_cifar100_data_aug(batch_size=128, num_workers=8,path=None,**kwargs):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(CIFAR10Policy())
    aug.append(transforms.ToTensor())
    aug.append(Cutout(n_holes=1, length=16))
    aug.append(
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    )

    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = CIFAR100(root=path,
                                train=True, download=True, transform=transform_train)
    test_dataset = CIFAR100(root=path,
                            train=False, download=False, transform=transform_test)

    #train_sampler = DistributedSampler(train_dataset)
    #val_sampler = DistributedSampler(val_dataset, round_up=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

def unpack_mix_param(args):
    mix_up = args['mix_up'] if 'mix_up' in args else False
    cut_mix = args['cut_mix'] if 'cut_mix' in args else False
    event_mix = args['event_mix'] if 'event_mix' in args else False
    beta = args['beta'] if 'beta' in args else 1.
    prob = args['prob'] if 'prob' in args else .5
    num = args['num'] if 'num' in args else 1
    num_classes = args['num_classes'] if 'num_classes' in args else 10
    noise = args['noise'] if 'noise' in args else 0.
    gaussian_n = args['gaussian_n'] if 'gaussian_n' in args else None
    return mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n

def get_cifar10dvs_data(batch_size=128, num_workers=8,path=None,**kwargs):
    """
    获取DVS CIFAR10数据
    http://journal.frontiersin.org/article/10.3389/fnins.2017.00309/full
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    root="/data/wupeixuan/"
    step=10
    size = kwargs['size'] if 'size' in kwargs else 48
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    train_dataset = tonic.datasets.CIFAR10DVS(os.path.join(root, 'DVS/DVS_Cifar10'), transform=train_transform)
    test_dataset = tonic.datasets.CIFAR10DVS(os.path.join(root, 'DVS/DVS_Cifar10'), transform=test_transform)
    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: torch.nn.functional.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        # lambda x: TemporalShift(x, .01),
        # lambda x: drop(x, 0.15),
        # lambda x: ShearX(x, 15),
        # lambda x: ShearY(x, 15),
        # lambda x: TranslateX(x, 0.225),
        # lambda x: TranslateY(x, 0.225),
        # lambda x: Rotate(x, 15),
        # lambda x: CutoutAbs(x, 0.25),
        # lambda x: CutoutTemporal(x, 0.25),
        # lambda x: GaussianBlur(x, 0.5),
        # lambda x: SaltAndPepperNoise(x, 0.1),
        # transforms.Normalize(DVSCIFAR10_MEAN_16, DVSCIFAR10_STD_16),
        transforms.RandomCrop(size, padding=size // 12),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: torch.nn.functional.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    if 'rand_aug' in kwargs.keys():
        if kwargs['rand_aug'] is True:
            n = kwargs['randaug_n']
            m = kwargs['randaug_m']
            # print('randaug', m, n)
            train_transform.transforms.insert(2, RandAugment(m=m, n=n))

    if 'temporal_flatten' in kwargs.keys():
        if kwargs['temporal_flatten'] is True:
            train_transform.transforms.insert(-1, lambda x: temporal_flatten(x))
            test_transform.transforms.insert(-1, lambda x: temporal_flatten(x))

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(root, 'DVS/DVS_Cifar10/train_cache_{}'.format(step)),
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(root, 'DVS/DVS_Cifar10/test_cache_{}'.format(step)),
                                     transform=test_transform)

    num_train = len(train_dataset)
    num_per_cls = num_train // 10
    indices_train, indices_test = [], []
    portion = kwargs['portion'] if 'portion' in kwargs else .9
    for i in range(10):
        indices_train.extend(
            list(range(i * num_per_cls, round(i * num_per_cls + num_per_cls * portion))))
        indices_test.extend(
            list(range(round(i * num_per_cls + num_per_cls * portion), (i + 1) * num_per_cls)))

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        # print('cut_mix', beta, prob, num, num_classes)
        train_dataset = CutMix(train_dataset,
                               beta=beta,
                               prob=prob,
                               num_mix=num,
                               num_class=num_classes,
                               indices=indices_train,
                               noise=noise)

    if event_mix:
        train_dataset = EventMix(train_dataset,
                                 beta=beta,
                                 prob=prob,
                                 num_mix=num,
                                 num_class=num_classes,
                                 indices=indices_train,
                                 noise=noise,
                                 gaussian_n=gaussian_n)

    if mix_up:
        train_dataset = MixUp(train_dataset,
                              beta=beta,
                              prob=prob,
                              num_mix=num,
                              num_class=num_classes,
                              indices=indices_train,
                              noise=noise)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
        pin_memory=True, drop_last=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        pin_memory=True, drop_last=False, num_workers=2
    )

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = get_cifar100_data(128)