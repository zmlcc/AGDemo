from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization
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
        para_shape = (1, num_channels, 1)  
        self.gamma = nn.Parameter(torch.ones(para_shape))
        self.beta = nn.Parameter(torch.zeros(para_shape))
        self.register_buffer('var_ema', torch.ones(para_shape))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                batch_ms = torch.mean(x.square(), dim=(0, 2), keepdim=True)
                self.var_ema.lerp_(batch_ms, self.momentum)
        else:
            batch_ms = self.var_ema

        batch_std = torch.sqrt(batch_ms + self.eps)

        return x / batch_std * self.gamma + self.beta


class StandardizedWeight(nn.Module):
    # weight.shape: (out_channels, in_channels, kernel_size)
    def forward(self, weight):
        eps = 1e-4
        fan_in = np.prod(weight.shape[1:])  # in_channels * kernel_size
        mean = torch.mean(weight, axis=[1, 2], keepdims=True)
        var = torch.var(weight, axis=[1, 2], keepdims=True)
        scale = torch.rsqrt((var * fan_in).clamp(min=eps))
        return (weight - mean) * scale

class StandardizedConv1D(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        *args,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, *args, **kwargs)

        register_parametrization(self, "weight", StandardizedWeight())



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


