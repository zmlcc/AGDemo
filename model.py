from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class ModelArgs:
    n_layers = 6



# ein notation

# b - batch
# c - feature channel
# s - sequence

# data shape: (batch_size, num_channels, seq_len)

class RMSBatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-05, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                batch_var = torch.var(x, dim=(0, 2), unbiased=False)
                self.running_var.lerp_(batch_var, self.momentum)
        else:
            batch_var = self.running_var

        batch_std = batch_var.clamp(min=self.eps).sqrt()[None,:,None]
        gamma = self.gamma[None,:,None]
        beta = self.beta[None,:,None]

        return x / batch_std * gamma + beta




class StandardizedConv1D(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )

        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1))
        self.register_buffer(
            "eps", torch.tensor(1e-4, requires_grad=False), persistent=False
        )
        self.register_buffer(
            "fan_in",
            torch.tensor(
                np.prod(self.weight.shape[1:]),
                dtype=self.weight.dtype,
                requires_grad=False,
            ),
            persistent=False,
        )

    def _standardized_weights(self):
        # self.weight.shape: (out_channels, in_channels, kernel_size)
        mean = torch.mean(self.weight, axis=[1, 2], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2], keepdims=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain

    def forward(self, x):
        weight = self._standardized_weights()
        return self._conv_forward(x, weight, self.bias)


class ConvBlock(nn.Module):
    def __init__(self, num_channels, width=5):
        super().__init__()
        self.num_channels = num_channels
        self.width = width

    def forward(self, x):
        x = RMSBatchNorm(x.shape[-1])(x)
        x = nn.GELU()(x)
        if self.width == 1:
            x = nn.Linear(x.shape[-1], self.num_channels)(x)
        else:
            x = StandardizedConv1D(x.shape[-1], self.num_channels, self.width)(x)
        return x


class ConvBlock222(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.norm = RMSBatchNorm(in_channels)
        self.actv = nn.GELU()
        if kernel_size == 1:
            self.conv = nn.Linear(in_channels, out_channels)
        else:
            self.conv = StandardizedConv1D(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.norm(x)
        x = self.actv(x)
        x = self.conv(x)
        return x


class DnaEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(4, 768, 15)
        self.conv1 = ConvBlock222(768, 768, 15)

    def forward(self, x):
        out = self.conv0(x)
        return out + self.conv1(out)
    
class DownresBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels + 128
        self.conv0 = ConvBlock222(in_channels, out_channels)
        self.conv1 = ConvBlock222(out_channels, out_channels)

    def forward(self, x):
        out = self.conv0(x)
        out = out + Pad ???
        return out + self.conv1(out)

class SequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownresBlock(768)


