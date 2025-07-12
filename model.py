from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization


@dataclass
class ModelArgs:
    n_down_blocks = 6


# ein notation

# b - batch
# c - feature channel
# s - sequence

# data shape: (batch_size, seq_len, num_channels)


class RMSBatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-05, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        para_shape = (1, 1, num_channels)
        self.gamma = nn.Parameter(torch.ones(para_shape))
        self.beta = nn.Parameter(torch.zeros(para_shape))
        self.register_buffer("var_ema", torch.ones(para_shape))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                batch_ms = torch.mean(x.square(), dim=(0, 1), keepdim=True)
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


class Conv1D(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        out = x.transpose(1, 2)  # (b, s, c) -> (b, c, s)
        out = super().forward(out)
        out.transpose_(1, 2)  # (b, c, s) -> (b, s, c)
        return out


class StandardizedConv1D(Conv1D):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, *args, **kwargs)

        register_parametrization(self, "weight", StandardizedWeight())


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()

        if kernel_size == 1:
            conv = nn.Linear(in_channels, out_channels)
        else:
            conv = StandardizedConv1D(
                in_channels, out_channels, kernel_size, padding=kernel_size // 2
            )

        self.net = nn.Sequential(RMSBatchNorm(in_channels), nn.GELU(), conv)

    def forward(self, x):
        return self.net(x)


class DnaEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = Conv1D(4, 768, 15, padding=15 // 2)
        self.conv1 = ConvBlock(768, 768)

    def forward(self, x):
        out = self.conv0(x)
        return out + self.conv1(out)


class DownresBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pad = 128
        out_channels = in_channels + self.pad
        self.conv0 = ConvBlock(in_channels, out_channels)
        self.conv1 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.conv0(x)
        out = out + F.pad(x, (0, self.pad))
        return out + self.conv1(out)


class SequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks = nn.ModuleList(
            [DnaEmbedder()]
            + [DownresBlock(768 + 128 * i) for i in range(1, ModelArgs.n_down_blocks)]
        )

    def forward(self, x):
        self.intermediates = []

        for block in self.down_blocks:
            x = block(x)
            self.intermediates.append(x)
            # Maxpool
        return x
