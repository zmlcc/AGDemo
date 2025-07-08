from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class ModelArgs:
    n_layers = 6

class RMSBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.offset = nn.Parameter(torch.zeros(num_features))

    def _norm(self, x):
        return x * torch.rsqrt(x.square().mean(0, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight + self.offset

class StandardizedConv1D(nn.Conv1d):
    def __init__(self,in_channels, out_channels, kernel_size):
        super().__init__(in_channels, out_channels, kernel_size)

        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(np.prod(self.weight.shape[1:]), requires_grad=False).type_as(self.weight), persistent=False)


    def _standardized_weights(self):
        mean = torch.mean(self.weight, axis=[1,2], keepdims=True)
        var = torch.var(self.weight, axis=[1,2], keepdims=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain
    
    def forward(self, x):
        weight = self._standardized_weights()
        return self._conv_forward(x, weight, self.bias)
        

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
