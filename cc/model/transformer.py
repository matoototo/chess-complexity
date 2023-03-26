import torch.nn as nn
from collections import OrderedDict

class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate, ffn_activation, input_channels = 17):
        super(TransformerModel, self).__init__()
        self.input_layer = TransformerInputLayer(input_channels, d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dff, dropout_rate, ffn_activation)
        self.head = ComplexityHead(d_model)

    def forward(self, inputs):
        x = self.input_layer(inputs)
        x = self.encoder(x)
        x = self.head(x)
        return x

class TransformerInputLayer(nn.Module):
    def __init__(self, input_channels, d_model):
        super(TransformerInputLayer, self).__init__()
        self.d_model = d_model
        self.conv = nn.Conv2d(input_channels, self.d_model, kernel_size=1)

    def forward(self, x):
        # BxCx8x8 -> Bx64xD
        batch_size = x.size(0)
        y = self.conv(x)
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(batch_size, 8*8, self.d_model)
        return y

class TransformerEncoder(nn.Sequential):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate, ffn_activation):
        super(TransformerEncoder, self).__init__(OrderedDict([
            (f"layer{i}", TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate, ffn_activation))
            for i in range(num_layers)
        ]))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate, ffn_activation):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            ffn_activation(),
            nn.Linear(dff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        y1 = self.mha(x, x, x, need_weights = False)
        y1 = self.ln1(y1 + x)
        y2 = self.ffn(y1)
        y2 = self.ln2(y2 + y1)
        return y2

class ComplexityHead(nn.Sequential):
    def __init__(self, d_model):
        super().__init__(OrderedDict([
            ("flatten", nn.Flatten()),
            ("lin1", nn.Linear(8*8*d_model, 1)),
            ("tanh", nn.Tanh()),
            ("lin2", nn.Linear(1, 1))
        ]))
