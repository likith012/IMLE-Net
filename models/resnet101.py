"""An implementation of ECGNet: Deep Network for Arrhythmia Classification

More details on the paper at https://ieeexplore.ieee.org/abstract/document/8438739

This file can also be imported as a module and contains the following functions:

    * conv3x3 - 3x3 convolution with padding
    * BasicBlock - Implementation of a single Residual block
    * Bottleneck - Implementation of a bottleneck Residual block
    * ResNet - Builds the Resnet model
    * resnet101 - Builds the Resnet-101 model
    
"""

import math
from typing import List

import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding

    Parameters
    ----------
    in_planes: int
        The number of input channels.
    out_planes: int
        The number of output channels.
    stride: int, optional
        The stride for 1D-convolution. (default: 1)

    Returns
    -------
    nn.Module
        The convolutional layer.
    """
    return nn.Conv1d(
        in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False
    )


class BasicBlock(nn.Module):
    """Implemetation of a single Residual block.

    Attributes
    ----------
    expansion: int
        The expansion factor for the block.
    stride: int, optional
        The stride for 1D-convolution. (default: 1)
    downsample: bool, optional
        If True, downsamples the input. (default: None)
    dropout: float
        The dropout probability.

    """

    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: bool = None
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Implementation of a bottleneck Residual block

    Attributes
    ----------
    expansion: int
        The expansion factor for the block.
    stride: int, optional
        The stride for 1D-convolution. (default: 1)
    downsample: bool, optional
        If True, downsamples the input. (default: None)

    """

    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: bool = None
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, bias=False, padding=3)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=11, stride=stride, padding=5, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=7, bias=False, padding=3)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Builds the Resnet model"""

    def __init__(self, block: int, layers: int, num_classes: int = 5) -> None:
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(
        self, block: int, planes: int, blocks: int, stride: int = 1
    ) -> List[nn.Module]:
        """Builds the Resnet layers

        Parameters
        ----------
         block: nn.Module
             The block to use.
         planes: int
             The number of input channels.
         blocks: int
             The number of blocks to use.
         stride: int, optional
             The stride for 1D-convolution. (default: 1)

         Returns
         -------
         nn.Module
             The output of Resnet layer.

        """

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet101(**kwargs) -> ResNet:
    """Builds the Resnet-101 model

    Returns
    -------
    nn.Module
        The output of Resnet-101 model.

    """

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model
