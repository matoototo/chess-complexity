import torch
import torch.nn as nn
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self, filters, blocks, head_neurons):
        super().__init__()
        self.args = (filters, blocks, head_neurons)
        self.input = InputBlock(15, filters)
        self.blocks = nn.Sequential(
            OrderedDict([(f"block{i}", Block(filters)) for i in range(blocks)])
        )
        self.output = ComplexityHead(filters, head_neurons)

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        x = self.output(x)
        return x


class Block(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.layers = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(filters, filters, 3, padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(filters)),
                ("relu1", nn.ReLU(inplace=True)),
                ("conv2", nn.Conv2d(filters, filters, 3, padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(filters))
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
                ("bn1", nn.BatchNorm2d(filters)),
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

