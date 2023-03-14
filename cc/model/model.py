import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init

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
