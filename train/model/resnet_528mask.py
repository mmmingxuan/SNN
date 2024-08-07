import torch
import torch.nn as nn
from ..node import functional
from ..node.LIFAct import LIFAct, LIFAct_withoutbn
import math
from einops import rearrange

# try:
#     from torchvision.models.utils import load_state_dict_from_url
# except ImportError:
#     from torchvision._internally_replaced_utils import load_state_dict_from_url
from .conv import conv3x3, conv1x1
from .basicblock import MultiStepBasicBlock
from .batchnorm import myBatchNorm2d, myBatchNorm3d
from .bottenleck import bottleneck


class MultiStepResNet18(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        datasets="imnet",
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=[False, False, False],
        norm_layer=None,
        T: int = None,
        multi_step_neuron: callable = None,
        multi_out=False,
        SE=False,
        multi_groups=1,
        **kwargs
    ):
        super().__init__()
        self.T = T
        self._norm_layer = norm_layer
        self.multi_out = multi_out
        self.inplanes = 64
        self.dilation = 1
        self.SE = SE
        self.datasets = datasets
        self.multi_groups = multi_groups
        self.groups = groups
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.base_width = width_per_group
        if "imnet" in datasets:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif "cifar100" in datasets or "CIFAR10" in datasets:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            if "CIFAR10" in datasets:
                num_classes = 10
            else:
                num_classes = 100
        elif "cifar10_dvs" in datasets:
            self.conv1 = nn.Conv2d(
                2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            num_classes = 10
        self.bn1 = norm_layer(self.inplanes)
        if "LIFAct" in str(multi_step_neuron):
            self.sn1 = multi_step_neuron(self.inplanes)
        else:
            self.sn1 = multi_step_neuron(**kwargs)
        # self.bn2 = norm_layer(self.inplanes)

        if "imnet" in datasets:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif (
            "cifar100" in datasets or "CIFAR10" in datasets or "cifar10_dvs" in datasets
        ):
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(
            block, 64, layers[0], multi_step_neuron=multi_step_neuron, **kwargs
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            multi_step_neuron=multi_step_neuron,
            **kwargs
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            multi_step_neuron=multi_step_neuron,
            **kwargs
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            multi_step_neuron=multi_step_neuron,
            **kwargs
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)

        if multi_out:
            self.bottleneck1_1 = bottleneck(
                64 * block.expansion,
                512 * block.expansion,
                kernel_size=8,
                node=multi_step_neuron,
                norm=norm_layer,
                groups=self.multi_groups,
                **kwargs
            )
            self.bottleneck2_1 = bottleneck(
                128 * block.expansion,
                512 * block.expansion,
                kernel_size=4,
                node=multi_step_neuron,
                norm=norm_layer,
                groups=self.multi_groups,
                **kwargs
            )
            self.bottleneck3_1 = bottleneck(
                256 * block.expansion,
                512 * block.expansion,
                kernel_size=2,
                node=multi_step_neuron,
                norm=norm_layer,
                groups=self.multi_groups,
                **kwargs
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if "LIFAct" in str(multi_step_neuron):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 1.0 / float(n))
                    m.bias.data.zero_()

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        multi_step_neuron: callable = None,
        **kwargs
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self._norm_layer == myBatchNorm3d:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    myBatchNorm3d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                multi_step_neuron,
                **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    multi_step_neuron=multi_step_neuron,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor, label=None, epoch=None):
        # See note [TorchScript super()]
        x_seq = None
        if x.dim() == 5:
            # x.shape = [T, N, C, H, W]
            if "cifar10_dvs" in self.datasets:
                x = x.permute(1, 0, 2, 3, 4)
            x_seq = functional.seq_to_ann_forward(x, [self.conv1])
            x_seq = self.bn1(x_seq)

        else:
            x = self.conv1(x)
            x.unsqueeze_(0)
            x_seq = x.repeat(self.T, 1, 1, 1, 1)
            x_seq = self.bn1(x_seq)

        x_seq = self.sn1(x_seq)
        x_seq = functional.seq_to_ann_forward(x_seq, self.maxpool)
        x_seq = self.layer1(x_seq)
        if self.multi_out:
            x1s = self.bottleneck1_1(x_seq)
            x1s = functional.seq_to_ann_forward(x1s, self.avgpool1)
            x1s = torch.flatten(x1s, 2)
            x1s = functional.seq_to_ann_forward(x1s, self.fc1)

        x_seq = self.layer2(x_seq)
        if self.multi_out:
            x2s = self.bottleneck2_1(x_seq)
            x2s = functional.seq_to_ann_forward(x2s, self.avgpool2)
            x2s = torch.flatten(x2s, 2)
            x2s = functional.seq_to_ann_forward(x2s, self.fc2)

        x_seq = self.layer3(x_seq)
        if self.multi_out:
            x3s = self.bottleneck3_1(x_seq)
            x3s = functional.seq_to_ann_forward(x3s, self.avgpool3)
            x3s = torch.flatten(x3s, 2)
            x3s = functional.seq_to_ann_forward(x3s, self.fc3)

        x_seq = self.layer4(x_seq)

        if self.training and self.SE:
            outs = []
            for i in range(x_seq.size(0)):
                if i == 0:
                    x_tmp = x_seq[i].detach()
                    x_single = x_seq[i]
                    x_single = self.avgpool(x_single)
                    x_single = torch.flatten(x_single, 1)
                    x_single = self.fc(x_single)
                    outs.append(x_single)

                    x_tmp = self.fc(x_tmp.view(-1, 512))
                    x_tmp = x_tmp.detach()
                    x_tmp = x_tmp.view(
                        x_seq.shape[1],
                        self.fc.out_features,
                        x_seq.shape[3],
                        x_seq.shape[3],
                    )  # 128, 100, 4, 4
                    mask = x_tmp[torch.arange(128), label]
                    mask = mask.detach()

                else:
                    x_tmp = x_seq[i].detach()
                    x_single = x_seq[i]

                    # 1. 1-mask use w 控制使用程度  w在0.6以下出现过拟合
                    # w = 0.6
                    # mask = functional.normalize_mask(mask)
                    # mask = w + (1 - w) * (1 - mask)

                    # scale = x_seq.shape[3] * x_seq.shape[4] / mask.sum(dim=(1, 2))
                    # scale = scale.unsqueeze(1).unsqueeze(2)
                    # mask = (mask * scale).unsqueeze(1)

                    # 1.1 1-mask use w 控制使用程度  使用softmax
                    # w = 0.8
                    # mask = functional.softmax_normalize_mask(mask)
                    # mask = w + (1 - w) * (1 - mask)

                    # scale = x_seq.shape[3] * x_seq.shape[4] / mask.sum(dim=(1, 2))
                    # scale = scale.unsqueeze(1).unsqueeze(2)
                    # mask = (mask * scale).unsqueeze(1)

                    # 2. 模仿Self-erasing, 进行背景、潜在区筛选   mask = functional.normalize_mask(mask)
                    # mask = functional.normalize_mask(mask)
                    # trinary_mask = torch.ones_like(mask)  # 初始化为1
                    # trinary_mask = torch.where(mask < 0.05, 0, trinary_mask)  # 小于0.05的赋值为0
                    # trinary_mask = torch.where(mask > 0.8, 0, trinary_mask)  # 大于0.7的赋值为0
                    # mask = trinary_mask.unsqueeze(1)

                    # 2.1 scale版 三元掩码   mask = functional.normalize_mask(mask)
                    # mask = functional.normalize_mask(mask)
                    # trinary_mask = torch.ones_like(mask)  # 初始化为1
                    # trinary_mask = torch.where(mask < 0.05, 0, trinary_mask)  # 小于0.05的赋值为0
                    # trinary_mask = torch.where(mask > 0.8, 0.5, trinary_mask)  # 大于0.7的赋值为0
                    # mask = trinary_mask.unsqueeze(1)
                    # scale = mask.shape[2] * mask.shape[3] / (mask.sum(dim=(2, 3))+ 1e-6)
                    # scale = scale.view(128, 1, 1, 1)
                    # mask = mask * scale

                    # 2.2 softmax计算mask   改为  mask = functional.softmax_normalize_mask(mask)
                    # mask = functional.softmax_normalize_mask(mask)
                    # trinary_mask = torch.ones_like(mask)  # 初始化为1
                    # trinary_mask = torch.where(mask < 0.01, 0, trinary_mask)  #
                    # trinary_mask = torch.where(mask > 0.8, 0.5, trinary_mask)  #
                    # mask = trinary_mask.unsqueeze(1)

                    # 2.3 scale + softmax计算mask   改为  mask = functional.softmax_normalize_mask(mask)
                    mask = functional.softmax_normalize_mask(mask)
                    trinary_mask = torch.ones_like(mask)  # 初始化为1
                    # trinary_mask = torch.where(mask < 0.01, 0, trinary_mask)  #
                    trinary_mask = torch.where(mask > 0.7, 0, trinary_mask)  #
                    mask = trinary_mask.unsqueeze(1)
                    scale = (
                        mask.shape[2] * mask.shape[3] / (mask.sum(dim=(2, 3)) + 1e-6)
                    )
                    scale = scale.view(128, 1, 1, 1)
                    mask = mask * scale

                    x_single = x_single * mask

                    # 2.4 每次计算mask时用的是上一时间步的 C×H×W 再乘上mask的结果
                    # x_tmp = x_single.detach()

                    x_single = self.avgpool(x_single)
                    x_single = torch.flatten(x_single, 1)
                    x_single = self.fc(x_single)
                    outs.append(x_single)

                    x_tmp = self.fc(x_tmp.view(-1, 512))
                    x_tmp = x_tmp.detach()
                    x_tmp = x_tmp.view(
                        x_seq.shape[1],
                        self.fc.out_features,
                        x_seq.shape[3],
                        x_seq.shape[3],
                    )  # 128, 100, 4, 4
                    mask = x_tmp[torch.arange(128), label]
                    mask = mask.detach()

            outs = torch.stack(outs, dim=0).unsqueeze(0)

            return outs

        else:
            x_seq = functional.seq_to_ann_forward(x_seq, self.avgpool)
            x_seq = torch.flatten(x_seq, 2)
            x_seq = functional.seq_to_ann_forward(x_seq, self.fc)
            if self.multi_out:
                return torch.stack([x1s, x2s, x3s, x_seq])
            else:
                return x_seq.unsqueeze(0)

    def forward(self, x, label=None, epoch=None):
        """
        :param x: the input with `shape=[N, C, H, W]` or `[*, N, C, H, W]`
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        return self._forward_impl(x, label, epoch)


class MultiStepResNet19(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        datasets="imnet",
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=[False, False, False],
        norm_layer=None,
        T: int = None,
        multi_step_neuron: callable = None,
        multi_out=False,
        SE=False,
        multi_groups=1,
        **kwargs
    ):
        super().__init__()
        self.T = T
        self._norm_layer = norm_layer
        self.multi_out = multi_out
        self.inplanes = 128
        self.dilation = 1
        self.SE = SE
        self.multi_groups = multi_groups
        self.datasets = datasets
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        if "imnet" in datasets:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif "cifar100" in datasets or "CIFAR10" in datasets:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            if "CIFAR10" in datasets:
                num_classes = 10
            else:
                num_classes = 100
        elif "cifar10_dvs" in datasets:
            self.conv1 = nn.Conv2d(
                2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            num_classes = 10
        self.bn1 = norm_layer(self.inplanes)
        if "LIFAct" in str(multi_step_neuron):
            self.sn1 = multi_step_neuron(self.inplanes)
        else:
            self.sn1 = multi_step_neuron(**kwargs)

        if "imnet" in datasets:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif (
            "cifar100" in datasets or "CIFAR10" in datasets or "cifar10_dvs" in datasets
        ):
            self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(
            block,
            128,
            layers[0],
            stride=1,
            multi_step_neuron=multi_step_neuron,
            **kwargs
        )
        self.layer2 = self._make_layer(
            block,
            256,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            multi_step_neuron=multi_step_neuron,
            **kwargs
        )
        self.layer3 = self._make_layer(
            block,
            512,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            multi_step_neuron=multi_step_neuron,
            **kwargs
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.tmp_conv1 = nn.Conv2d(
            512 * block.expansion,
            num_classes * self.multi_groups,
            kernel_size=1,
            groups=self.multi_groups,
        )
        self.tmp_conv2 = nn.Conv2d(
            512 * block.expansion,
            num_classes * self.multi_groups,
            kernel_size=1,
            groups=self.multi_groups,
        )

        if multi_out:
            self.bottleneck1_1 = bottleneck(
                128 * block.expansion,
                512 * block.expansion,
                kernel_size=4,
                node=multi_step_neuron,
                norm=norm_layer,
                groups=self.multi_groups,
                **kwargs
            )
            self.bottleneck2_1 = bottleneck(
                256 * block.expansion,
                512 * block.expansion,
                kernel_size=2,
                node=multi_step_neuron,
                norm=norm_layer,
                groups=self.multi_groups,
                **kwargs
            )
            # self.bottleneck3_1 = bottleneck(512 * block.expansion, 512 * block.expansion,kernel_size=2, node=multi_step_neuron,norm=norm_layer,**kwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             if hasattr(m.bn3, 'weight'):
        #                 nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             if hasattr(m.bn2, 'weight'):
        #                 nn.init.constant_(m.bn2.weight, 0)

        if "LIFAct" in str(multi_step_neuron):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 1.0 / float(n))
                    m.bias.data.zero_()

        # self.mask1_max1 = []
        # self.mask1_max2 = []
        # self.mask1_max3 = []

        # self.mask2_max1 = []
        # self.mask2_max2 = []
        # self.mask2_max3 = []

        # self.mask3_max1 = []
        # self.mask3_max2 = []
        # self.mask3_max3 = []

        self.mask1_q2 = []
        self.mask1_q1 = []
        self.mask1_q3 = []
        self.mask2_q2 = []
        self.mask2_q1 = []
        self.mask2_q3 = []
        self.mask3_q2 = []
        self.mask3_q1 = []
        self.mask3_q3 = []

        self.mask1_activ = []
        self.mask1_backg = []

        self.mask2_activ = []
        self.mask2_backg = []

        self.mask3_activ = []
        self.mask3_backg = []

        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []

    def get_top_three_values(self, mask, i):

        # top_values, _ = torch.topk(mask.view(-1), 3)
        median = torch.quantile(mask, 0.5).item()
        q1 = torch.quantile(mask, 0.25).item()
        q3 = torch.quantile(mask, 0.75).item()

        if i == 1:
            # self.mask1_max1.append(top_values[0].item())
            # self.mask1_max2.append(top_values[1].item())
            # self.mask1_max3.append(top_values[2].item())
            self.mask1_q2.append(median)
            self.mask1_q1.append(q1)
            self.mask1_q3.append(q3)
        elif i == 2:
            # self.mask2_max1.append(top_values[0].item())
            # self.mask2_max2.append(top_values[1].item())
            # self.mask2_max3.append(top_values[2].item())
            self.mask2_q2.append(median)
            self.mask2_q1.append(q1)
            self.mask2_q3.append(q3)
        elif i == 3:
            # self.mask3_max1.append(top_values[0].item())
            # self.mask3_max2.append(top_values[1].item())
            # self.mask3_max3.append(top_values[2].item())
            self.mask3_q2.append(median)
            self.mask3_q1.append(q1)
            self.mask3_q3.append(q3)

    def reset_mask(self):
        # self.mask1_max1 = []
        # self.mask1_max2 = []
        # self.mask1_max3 = []

        # self.mask2_max1 = []
        # self.mask2_max2 = []
        # self.mask2_max3 = []

        # self.mask3_max1 = []
        # self.mask3_max2 = []
        # self.mask3_max3 = []

        self.mask1_q2 = []
        self.mask1_q1 = []
        self.mask1_q3 = []
        self.mask2_q2 = []
        self.mask2_q1 = []
        self.mask2_q3 = []
        self.mask3_q2 = []
        self.mask3_q1 = []
        self.mask3_q3 = []

        self.mask1_activ = []
        self.mask1_backg = []

        self.mask2_activ = []
        self.mask2_backg = []

        self.mask3_activ = []
        self.mask3_backg = []

        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        multi_step_neuron: callable = None,
        **kwargs
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self._norm_layer == myBatchNorm3d:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    myBatchNorm3d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                multi_step_neuron,
                **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    multi_step_neuron=multi_step_neuron,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor, label=None, epoch=None):
        # See note [TorchScript super()]

        # ------仅用于调试--------------------
        # x = torch.load('fig/normalized_cat_tensor.pt').unsqueeze(0).to(x.device, non_blocking=True)
        # --------------------------

        x_seq = None
        if x.dim() == 5:
            # x.shape = [T, N, C, H, W]
            if "cifar10_dvs" in self.datasets:
                x = x.permute(1, 0, 2, 3, 4)
            x_seq = functional.seq_to_ann_forward(x, [self.conv1])
            x_seq = self.bn1(x_seq)
        else:
            assert (
                self.T is not None
            ), "When x.shape is [N, C, H, W], self.T can not be None."
            # x.shape = [N, C, H, W]
            x = self.conv1(x)
            x.unsqueeze_(0)
            x_seq = x.repeat(self.T, 1, 1, 1, 1)
            x_seq = self.bn1(x_seq)

        x_seq = self.sn1(x_seq)
        x_seq = functional.seq_to_ann_forward(x_seq, self.maxpool)
        x_seq = self.layer1(x_seq)

        if self.multi_out:
            x1s = self.bottleneck1_1(x_seq)
            if self.training and self.SE:
                outs = []
                for i in range(x1s.size(0)):
                    if i == 0:
                        x_tmp = x1s[i].detach()
                        x_single = x1s[i]
                        x_single = self.avgpool1(x_single)
                        x_single = torch.flatten(x_single, 1)
                        x_single = self.fc1(x_single)
                        outs.append(x_single)

                        x_tmp = self.fc1(x_tmp.view(-1, 512))
                        x_tmp = x_tmp.detach()
                        x_tmp = x_tmp.view(
                            x1s.shape[1],
                            self.fc1.out_features,
                            x1s.shape[3],
                            x1s.shape[3],
                        )  # 128, 100, 8, 8
                        mask = x_tmp[torch.arange(128), label]
                        mask = mask.detach()
                    else:
                        x_tmp = x1s[i].detach()
                        x_single = x1s[i]
                        w = 0
                        mask = functional.softmax_normalize_mask(mask)
                        mask = w + (1 - w) * (1 - mask)

                        scale = x1s.shape[3] * x1s.shape[4] / mask.sum(dim=(1, 2))
                        scale = scale.unsqueeze(1).unsqueeze(2)
                        mask = (mask * scale).unsqueeze(1)

                        x_single = x_single * mask

                        # 2.4 每次计算mask时用的是上一时间步的 C×H×W 再乘上mask的结果
                        # x_tmp = x_single.detach()

                        x_single = self.avgpool1(x_single)
                        x_single = torch.flatten(x_single, 1)
                        x_single = self.fc1(x_single)
                        outs.append(x_single)

                        x_tmp = self.fc1(x_tmp.view(-1, 512))
                        x_tmp = x_tmp.detach()
                        x_tmp = x_tmp.view(
                            x1s.shape[1],
                            self.fc1.out_features,
                            x1s.shape[3],
                            x1s.shape[3],
                        )  # 128, 100, 4, 4
                        mask = x_tmp[torch.arange(128), label]
                        mask = mask.detach()

                outs = torch.stack(outs, dim=0)

                x1s = outs
            else:
                x1s = functional.seq_to_ann_forward(x1s, self.avgpool1)
                # x1s=torch.flatten(x1s, 2)
                # x1s=functional.seq_to_ann_forward(x1s, self.fc1)
                x1s = functional.seq_to_ann_forward(x1s, self.tmp_conv1)
                x1s = torch.squeeze(
                    torch.cat(torch.chunk(x1s, self.multi_groups, dim=2), dim=0),
                    dim=(-1, -2),
                )  # * 分组卷积的精华，用1*1卷积代替gruops个全连接层
        x_seq = self.layer2(x_seq)

        if self.multi_out:
            x2s = self.bottleneck2_1(x_seq)
            if self.training and self.SE:
                outs = []
                for i in range(x2s.size(0)):
                    if i == 0:
                        x_tmp = x2s[i].detach()
                        x_single = x2s[i]
                        x_single = self.avgpool2(x_single)
                        x_single = torch.flatten(x_single, 1)
                        x_single = self.fc2(x_single)
                        outs.append(x_single)

                        x_tmp = self.fc2(x_tmp.view(-1, 512))
                        x_tmp = x_tmp.detach()
                        x_tmp = x_tmp.view(
                            x2s.shape[1],
                            self.fc2.out_features,
                            x2s.shape[3],
                            x2s.shape[3],
                        )  # 128, 100, 8, 8
                        mask = x_tmp[torch.arange(128), label]
                        mask = mask.detach()
                    else:
                        x_tmp = x2s[i].detach()
                        x_single = x2s[i]
                        w = 0
                        mask = functional.softmax_normalize_mask(mask)
                        mask = w + (1 - w) * (1 - mask)

                        scale = x2s.shape[3] * x2s.shape[4] / mask.sum(dim=(1, 2))
                        scale = scale.unsqueeze(1).unsqueeze(2)
                        mask = (mask * scale).unsqueeze(1)

                        x_single = x_single * mask

                        # 2.4 每次计算mask时用的是上一时间步的 C×H×W 再乘上mask的结果
                        # x_tmp = x_single.detach()

                        x_single = self.avgpool2(x_single)
                        x_single = torch.flatten(x_single, 1)
                        x_single = self.fc2(x_single)
                        outs.append(x_single)

                        x_tmp = self.fc2(x_tmp.view(-1, 512))
                        x_tmp = x_tmp.detach()
                        x_tmp = x_tmp.view(
                            x2s.shape[1],
                            self.fc2.out_features,
                            x2s.shape[3],
                            x2s.shape[3],
                        )  # 128, 100, 4, 4
                        mask = x_tmp[torch.arange(128), label]
                        mask = mask.detach()

                outs = torch.stack(outs, dim=0)

                x2s = outs

            else:
                x2s = functional.seq_to_ann_forward(x2s, self.avgpool2)
                # x2s=torch.flatten(x2s, 2)
                # x2s=functional.seq_to_ann_forward(x2s, self.fc2)
                x2s = functional.seq_to_ann_forward(x2s, self.tmp_conv2)
                x2s = torch.squeeze(
                    torch.cat(torch.chunk(x2s, self.multi_groups, dim=2), dim=0),
                    dim=(-1, -2),
                )
        x_seq = self.layer3(x_seq)

        if self.training and self.SE:
            outs = []
            masks_tmp = []
            masks = []
            for i in range(x_seq.size(0)):
                x_tmp = x_seq[i].detach()
                x_tmp = (
                    self.fc(
                        x_tmp.permute(0, 2, 3, 1).contiguous().view(128 * 8 * 8, 512)
                    )
                    .view(128, 8, 8, 100)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                mask = x_tmp
                # x_tmp = torch.nn.functional.softmax(x_tmp, dim=1)  # softmax应用在每个8x8像素点上的100维向量
                # mask = x_tmp[torch.arange(len(label)), label]

                # mask = functional.softmax_normalize_mask(mask)

                mask = mask.detach()
                masks_tmp.append(mask)

            masks_tmp = torch.stack(masks_tmp)

            # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            # 1-其他三个的平均值
            # for i in range(masks_tmp.size(0)):
            #     others_mean = torch.mean(torch.cat([masks_tmp[:i], masks_tmp[i+1:]]), dim=0)
            #     w = 0.5
            #     mask = w + (1 - others_mean)*(1 - w)
            #     masks.append(mask)

            # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            # 每一步关注之前所有步关注不到的，该时间步之前所有时间步的1-mean，不包含自己，所以第0个mask是全为1的
            # masks = [torch.ones_like(masks_tmp[0])]
            # for i in range(1, masks_tmp.size(0)):
            #     others_mean = torch.mean(masks_tmp[:i], dim=0)
            #     w = 0.5
            #     mask = w + (1 - others_mean)*(1 - w)
            #     masks.append(mask)

            # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            # 每一步关注之前步关注不到的 ,进行掩码，不包含自己，所以第0个mask是全为1的
            # masks = []
            # mask = torch.ones_like(masks_tmp[0])
            # masks.append(mask)
            # th = 0.8
            # for i in range(1, masks_tmp.size(0)):
            # #     # 注释掉代表mask会继承
            #     # mask = torch.ones_like(masks_tmp[0])

            # #     # 动态阈值，阈值为均值＋标准差
            # #     # th = functional.calculate_th(masks_tmp[i-1]).to(mask.device)
            # #     # mask = torch.where(masks_tmp[i-1] > th.unsqueeze(1).unsqueeze(2), 0, mask)

            #     # 固定阈值
            #     mask = torch.where(masks_tmp[i-1] > th, 0, mask)

            # #     # 掩掉最大的n个
            # #     # n = 20
            # #     # mask = functional.mask_top_n(masks_tmp[i-1], n)

            #     masks.append(mask)
            #     if i==1:
            #         self.mask1_activ.append(mask.sum().item()/128)
            #     elif i==2:
            #         self.mask2_activ.append(mask.sum().item()/128)
            #     elif i==3:
            #         self.mask3_activ.append(mask.sum().item()/128)

            # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            # One To ALL 该时间步之外的所有时间步过完FC平均再Softmax,不包含自己\
            # 所以每一步都有mask(即使是第一步)\
            # 所以要记得把之前的softmax注释掉和挑选label那一行，这里操作的masks_tmp是原始过完FC的
            # for i in range(masks_tmp.size(0)):
            #     others_mean = torch.mean(torch.cat([masks_tmp[:i], masks_tmp[i+1:]]), dim=0)
            #     others_soft = torch.nn.functional.softmax(others_mean, dim=1)
            #     others_mask = others_soft[torch.arange(len(label)), label]

            #     # 固定阈值
            #     # th = 0.4
            #     # mask = torch.ones_like(others_mask)  # 注释掉此行代表继承mask
            #     # mask = torch.where(others_mask > th, 0, mask)

            #     # 动态阈值 + 选和固定阈值中相比更大的那个
            #     w = -0.4
            #     mask = torch.ones_like(others_mask)
            #     th = functional.calculate_th(others_mask, w).to(mask.device)
            #     y = 0.5
            #     y_tensor = torch.tensor(y, device=th.device)
            #     th = torch.where(th < y_tensor, y_tensor, th)
            #     mask = torch.where(others_mask > th.unsqueeze(1).unsqueeze(2), 0, mask)

            #     masks.append(mask)
            #     if i==1:
            #         self.mask1_activ.append(mask.sum().item()/128)
            #     elif i==2:
            #         self.mask2_activ.append(mask.sum().item()/128)
            #     elif i==3:
            #         self.mask3_activ.append(mask.sum().item()/128)

            # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            # 每一步关注之前所有步关注不到的，该时间步之前所有时间步的过完FC平均再Softmax，不包含自己，所以第0个mask是全为1的
            # 所以要记得把之前的softmax注释掉和挑选label那一行，这里操作的masks_tmp是原始过完FC的
            # 抹除阈值以上的像素点
            masks = []
            # mask = torch.ones_like(masks_tmp[0])
            mask = torch.ones(128, 8, 8).to(masks_tmp[0].device)
            masks.append(mask)
            for i in range(1, masks_tmp.size(0)):
                others_mean = torch.mean(masks_tmp[:i], dim=0)
                others_soft = torch.nn.functional.softmax(others_mean, dim=1)
                others_mask = others_soft[torch.arange(len(label)), label]

                # 固定阈值
                # th = 0.4
                # mask = torch.ones_like(others_mask)  # 注释掉此行代表继承mask
                # mask = torch.where(others_mask > th, 0, mask)

                # 动态阈值 + 选和固定阈值中相比更大的那个
                w = -0.4
                mask = torch.ones_like(others_mask)
                th = functional.calculate_th(others_mask, w).to(mask.device)
                y = 0.5
                y_tensor = torch.tensor(y, device=th.device)
                th = torch.where(th < y_tensor, y_tensor, th)
                mask = torch.where(others_mask > th.unsqueeze(1).unsqueeze(2), 0, mask)

                masks.append(mask)
                if i == 1:
                    self.mask1_activ.append(mask.sum().item() / 128)
                elif i == 2:
                    self.mask2_activ.append(mask.sum().item() / 128)
                elif i == 3:
                    self.mask3_activ.append(mask.sum().item() / 128)

            # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            # 每一步关注之前所有步关注不到的，该时间步之前所有时间步的过完FC平均再Softmax，不包含自己，所以第0个mask是全为1的
            # 所以要记得把之前的softmax注释掉，这里操作的masks_tmp是原始过完FC的
            # 这个是固定抹除每个mask一定数量的最大值的像素点
            # masks = []
            # mask = torch.ones_like(masks_tmp[0])
            # masks.append(mask)
            # n = 20
            # for i in range(1, masks_tmp.size(0)):
            #     others_mean = torch.mean(masks_tmp[:i], dim=0)
            #     others_soft = torch.nn.functional.softmax(others_mean, dim=1)
            #     mask = functional.mask_top_n(others_soft, n)
            #     masks.append(mask)
            #     if i==1:
            #         self.mask1_activ.append(mask.sum().item()/128)
            #     elif i==2:
            #         self.mask2_activ.append(mask.sum().item()/128)
            #     elif i==3:
            #         self.mask3_activ.append(mask.sum().item()/128)

            # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
            # 前面为需要对100维mask进行softmax，8*8维度上先不考虑
            # 方法是想通过第0个时间步的预测结果，对每一个8*8的mask里的值排序后\
            # 分为等分的三个时间步的mask，这三个mask分别用于后面三个时间步
            # 这里有几个尝试，顺序问题，是让后面的时间步学习背景还是学习最高的那一份mask
            # 答: 目前来看前面的时间步学概率高的那份比较好(前面的时间步学潜在，后面学背景)
            # 其次是scale问题，还需不需要对mask进行scale操作
            # masks = []
            # mask = torch.ones_like(masks_tmp[0])
            # masks.append(mask)
            # mask1, mask2, mask3 = functional.calculate_mask(masks_tmp[0])
            # masks.append(mask3)
            # masks.append(mask2)
            # masks.append(mask1)
            # self.mask1_activ.append(mask1.sum().item()/128)
            # self.mask2_activ.append(mask2.sum().item()/128)
            # self.mask3_activ.append(mask3.sum().item()/128)

            # -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

            for i in range(x_seq.size(0)):
                x_single = x_seq[i]
                mask_sum = masks[i].sum(dim=(1, 2))
                mask_sum[mask_sum == 0] = 64  # 如果mask_sum为0，则设置为64
                scale = x_seq.shape[3] * x_seq.shape[4] / mask_sum
                scale = scale.unsqueeze(1).unsqueeze(2)
                mask = (masks[i] * scale).unsqueeze(1)
                x_single = x_single * mask
                x_single = self.avgpool(x_single)
                x_single = torch.flatten(x_single, 1)
                x_single = self.fc(x_single)
                outs.append(x_single)

            if self.multi_out:
                outs = torch.stack(outs, dim=0)
                x_seq = outs
            else:
                # self.p0.append(torch.mean(torch.nn.functional.softmax(outs[0], dim=1)[torch.arange(len(label)), label],dim=0).item())
                # self.p1.append(torch.mean(torch.nn.functional.softmax(outs[1], dim=1)[torch.arange(len(label)), label],dim=0).item())
                # self.p2.append(torch.mean(torch.nn.functional.softmax(outs[2], dim=1)[torch.arange(len(label)), label],dim=0).item())
                # self.p3.append(torch.mean(torch.nn.functional.softmax(outs[3], dim=1)[torch.arange(len(label)), label],dim=0).item())

                outs = torch.stack(outs, dim=0).unsqueeze(0)

                return outs

        else:
            x_seq = functional.seq_to_ann_forward(x_seq, self.avgpool)
            x_seq = torch.flatten(x_seq, 2)
            x_seq = functional.seq_to_ann_forward(x_seq, self.fc)

        if self.multi_out:
            # return torch.stack([x1s,x2s,x_seq])
            return torch.cat((x1s, x2s, x_seq), dim=0)
        else:
            # self.p0.append(torch.mean(torch.nn.functional.softmax(x_seq[0], dim=1)[torch.arange(len(label)), label],dim=0).item())
            # self.p1.append(torch.mean(torch.nn.functional.softmax(x_seq[1], dim=1)[torch.arange(len(label)), label],dim=0).item())
            # self.p2.append(torch.mean(torch.nn.functional.softmax(x_seq[2], dim=1)[torch.arange(len(label)), label],dim=0).item())
            # self.p3.append(torch.mean(torch.nn.functional.softmax(x_seq[3], dim=1)[torch.arange(len(label)), label],dim=0).item())
            return x_seq.unsqueeze(0)

    def forward(self, x, label=None, epoch=None):
        """
        :param x: the input with `shape=[N, C, H, W]` or `[*, N, C, H, W]`
        :type x: torch.Tensor
        :return: output
        :rtype: torch.Tensor
        """
        return self._forward_impl(x, label, epoch)


def _multi_step_resnet(
    arch,
    block,
    layers,
    pretrained,
    progress,
    T,
    multi_step_neuron,
    norm_layer,
    **kwargs
):

    print("arch---------------", arch)
    if arch == "resnet19":
        model = MultiStepResNet19(
            block,
            layers,
            T=T,
            multi_step_neuron=multi_step_neuron,
            norm_layer=norm_layer,
            **kwargs
        )
    else:
        model = MultiStepResNet18(
            block,
            layers,
            T=T,
            multi_step_neuron=multi_step_neuron,
            norm_layer=norm_layer,
            **kwargs
        )
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def multi_step_resnet18(
    pretrained=False,
    progress=True,
    T: int = None,
    multi_step_neuron: callable = None,
    norm_layer=myBatchNorm2d,
    **kwargs
):

    return _multi_step_resnet(
        "resnet18",
        MultiStepBasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        T,
        multi_step_neuron,
        norm_layer,
        **kwargs
    )


def multi_step_resnet19(
    pretrained=False,
    progress=True,
    T: int = None,
    multi_step_neuron: callable = None,
    norm_layer=myBatchNorm2d,
    **kwargs
):

    return _multi_step_resnet(
        "resnet19",
        MultiStepBasicBlock,
        [3, 3, 2],
        pretrained,
        progress,
        T,
        multi_step_neuron,
        norm_layer,
        **kwargs
    )
