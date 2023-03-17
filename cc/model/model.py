import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init

import numpy as np

class Model(nn.Module):
    def __init__(self, filters, blocks, head_neurons, head_v2 = False, use_se = False, se_ratio = 8, head_filters = 1, activation = nn.Mish):
        super().__init__()
        self.args = (filters, blocks, head_neurons, head_v2, use_se, se_ratio, head_filters, activation)
        self.input = InputBlock(17, filters, activation)
        self.blocks = nn.Sequential(
            OrderedDict([(f"block{i}", Block(filters, use_se, se_ratio, activation)) for i in range(blocks)])
        )
        if head_v2: self.output = ComplexityHeadV2(filters, head_neurons, head_filters, activation)
        else: self.output = ComplexityHead(filters, head_neurons, activation)
        self.reset_parameters()

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = self.output(x)
        return x

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                init.ones_(module.weight)
                init.zeros_(module.bias)

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio, activation):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(channels, channels // ratio)
        self.act = activation(inplace=True)
        self.lin2 = nn.Linear(channels // ratio, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.pool(x).view(batch, channels)
        y = self.lin1(y)
        y = self.act(y)
        y = self.lin2(y).view(batch, channels, 1, 1)
        scale = torch.sigmoid(y)
        return x * scale

class Block(nn.Module):
    def __init__(self, filters, use_se, se_ratio, activation):
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(filters, filters, 3, padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(filters, affine=True)),
                ("act1", activation(inplace=True)),
                ("conv2", nn.Conv2d(filters, filters, 3, padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(filters, affine=True))
            ])
        )
        if use_se:
            se = SqueezeExcitation(filters, se_ratio, activation)
            self.layers.add_module("se", se)
        self.act2 = activation(inplace=True)

    def forward(self, x):
        y = self.layers(x)
        y += x
        y = self.act2(y)
        return y

class InputBlock(nn.Sequential):
    def __init__(self, planes, filters, activation):
        super().__init__(
            OrderedDict([
                ("conv1", nn.Conv2d(planes, filters, 3, padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(filters, affine=True)),
                ("act1", activation(inplace=True)),
            ])
        )


class ComplexityHead(nn.Sequential):
    def __init__(self, filters, neurons, activation):
        super().__init__(OrderedDict([
            ('flatten', nn.Flatten()),
            ('lin1', nn.Linear(filters * 64, neurons)),
            ('act1', activation(inplace=True)),
            ('lin2', nn.Linear(neurons, 1))
        ]))

class ComplexityHeadV2(nn.Sequential):
    def __init__(self, filters, neurons, head_filters, activation):
        super().__init__(OrderedDict([
            ("conv1", nn.Conv2d(filters, head_filters, 1, bias=False)),
            ("bn1", nn.BatchNorm2d(head_filters, affine=True)),
            ("act1", activation(inplace=True)),
            ('flatten', nn.Flatten()),
            ('lin1', nn.Linear(head_filters * 64, neurons)),
            ('act2', activation(inplace=True)),
            ('lin2', nn.Linear(neurons, 1)),
            ("tanh", nn.Tanh()),
            ("lin3", nn.Linear(1, 1))
        ]))


class FixupInputBlock(nn.Module):
    def __init__(self, planes, filters, use_se, se_ratio, activation):
        super().__init__()

        self.filters = filters

        self.conv1 = nn.Conv2d(planes, filters, 3, padding=1, bias=False)
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.act1 = activation(inplace=True)

        if use_se:
            self.se = SqueezeExcitation(filters, se_ratio, activation)

    def forward(self, x):

        y = self.conv1(x)
        y = y + self.bias1a
        y = self.act1(y)

        if hasattr(self, "se"): # ?
            y = self.se(y)

        return y


class FixupBlock(nn.Module):
    def __init__(self, filters, use_se, se_ratio, activation):
        super().__init__()

        self.filters = filters

        self.bias1b = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.act1 = activation(inplace=True)

        self.bias2b = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.multiplier = nn.Parameter(torch.ones(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.act2 = activation(inplace=True)

        if use_se:
            self.se = SqueezeExcitation(filters, se_ratio, activation)

    def forward(self, x):

        y = self.conv1(x + self.bias1b)
        y = self.act1(y + self.bias1a)

        y = self.conv2(y + self.bias2b)
        y = self.act2(self.multiplier*y + self.bias2a + x)

        if hasattr(self, "se"): # ?
            y = self.se(y)

        return y

    def fixup_initialization(self, n_layers):
        std = np.sqrt(2 / (self.filters * 3 * 3)) / np.sqrt(n_layers)
        nn.init.normal_(self.conv1.weight, mean = 0, std = std)
        nn.init.constant_(self.conv2.weight, 0)


class FixupComplexityHeadV2(nn.Module):
    def __init__(self, filters, neurons, head_filters, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, head_filters, 1, bias=False)
        self.act1 = activation(inplace=True)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(head_filters * 64, neurons)
        self.act2 = activation(inplace=True)
        self.lin2 = nn.Linear(neurons, 1)
        self.tanh = nn.Tanh()
        self.lin3 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.act2(x)
        x = self.lin2(x)
        x = self.tanh(x)
        x = self.lin3(x)
        return x

    def fixup_initialization(self, n_layers):
        std = np.sqrt(2 / (self.conv1.out_channels * 3 * 3)) / np.sqrt(n_layers)
        nn.init.normal_(self.conv1.weight, mean = 0, std = std)


class FixupModel(nn.Module):
    def __init__(self, filters, blocks, head_neurons, head_v2 = False, use_se = False, se_ratio = 8, head_filters = 1, activation = nn.Mish):
        super().__init__()
        self.args = (filters, blocks, head_neurons, head_v2, use_se, se_ratio, head_filters, activation)
        self.input = FixupInputBlock(17, filters, use_se, se_ratio, activation)
        self.blocks = nn.Sequential(
            OrderedDict([(f"block{i}", FixupBlock(filters, use_se, se_ratio, activation)) for i in range(blocks)])
        )
        if head_v2: self.output = FixupComplexityHeadV2(filters, head_neurons, head_filters, activation)
        else: raise NotImplementedError("Only head v2 is implemented")
        self.reset_parameters()

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = self.output(x)
        return x

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                init.ones_(module.weight)
                init.zeros_(module.bias)

        for block in self.blocks:
            block.fixup_initialization(self.args[1])
        self.output.fixup_initialization(self.args[1])
