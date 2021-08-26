import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, filters, blocks, head_neurons, head_v2 = False):
        super().__init__()
        self.args = (filters, blocks, head_neurons, head_v2)
        self.input = InputBlock(16, filters)
        self.blocks = nn.Sequential(
            OrderedDict([(f"block{i}", Block(filters)) for i in range(blocks)])
        )
        if head_v2: self.output = ComplexityHeadV2(filters, head_neurons)
        else: self.output = ComplexityHead(filters, head_neurons)
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


class Block(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(filters, filters, 3, padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(filters, affine=True)),
                ("relu1", nn.ReLU(inplace=True)),
                ("conv2", nn.Conv2d(filters, filters, 3, padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(filters, affine=True))
            ])
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.layers(x)
        y += x
        y = self.relu2(y)
        return y

class InputBlock(nn.Sequential):
    def __init__(self, planes, filters):
        super().__init__(
            OrderedDict([
                ("conv1", nn.Conv2d(planes, filters, 3, padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(filters, affine=True)),
                ("relu1", nn.ReLU(inplace=True)),
            ])
        )


class ComplexityHead(nn.Sequential):
    def __init__(self, filters, neurons):
        super().__init__(OrderedDict([
            ('flatten', nn.Flatten()),
            ('lin1', nn.Linear(filters * 64, neurons)),
            ('relu1', nn.ReLU(inplace=True)),
            ('lin2', nn.Linear(neurons, 1))
        ]))

class ComplexityHeadV2(nn.Sequential):
    def __init__(self, filters, neurons):
        super().__init__(OrderedDict([
            ("conv1", nn.Conv2d(filters, 1, 1, bias=False)),
            ("bn1", nn.BatchNorm2d(1, affine=True)),
            ("relu1", nn.ReLU(inplace=True)),
            ('flatten', nn.Flatten()),
            ('lin1', nn.Linear(64, neurons)),
            ('relu1', nn.ReLU(inplace=True)),
            ('lin2', nn.Linear(neurons, 1))
        ]))
