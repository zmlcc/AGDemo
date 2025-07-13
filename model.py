from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from einops.layers.torch import Rearrange, Reduce
from einops import repeat, rearrange


@dataclass
class ModelArgs:
    n_down_blocks = 6


# ein notation

# b - batch_size
# c - num_channels
# s - seq_len
# p - pairwise_seq_len
# f - pairwise_channels

# data shape: (batch_size, seq_len, num_channels)


class RMSBatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-05, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.register_buffer("var_ema", torch.ones(num_channels))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                x_reshaped = x.view(-1, x.shape[-1])
                batch_ms = torch.mean(x_reshaped.square(), dim=0)
                self.var_ema.lerp_(batch_ms, self.momentum)
        else:
            batch_ms = self.var_ema

        x_norm = x * torch.rsqrt(batch_ms + self.eps)

        return x_norm * self.gamma + self.beta


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


class MlpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels * 2
        dropout_rate = 0.3
        self.net = nn.Sequential(
            RMSBatchNorm(in_channels),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_channels, out_channels),
            RMSBatchNorm(out_channels),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBiasBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            RMSBatchNorm(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, 8, bias=False),
            Rearrange("b p P h -> b h p P"),
        )

    # data shape: (b p P f) -> (b 8 s S)
    def forward(self, x):
        out = self.net(x)
        return repeat(out, "b h p P -> b h (p 8) (P 8)")


class RoPE(nn.Module):
    def __init__(self, dim, max_positions, positions=None):
        super().__init__()
        positions = positions or torch.arange(max_positions)
        num_freq = dim // 2
        freqs = 1.0 / (
            torch.arange(num_freq)
            + torch.logspace(0, np.log10(max_positions - num_freq + 1), num_freq)
        )
        theta = torch.outer(positions, freqs)
        theta = theta.repeat_interleave(2, dim=1)
        self.register_buffer("cos", theta.cos())
        self.register_buffer("sin", theta.sin())

    def forward(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * self.cos + x_rotated * self.sin


class MhaBlock(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        q_heads = 8
        kv_heads = 1
        q_channels = 128
        k_channels = 128
        v_channels = 192

        self.norm0 = RMSBatchNorm(in_channels)

        self.q_proj = nn.Sequential(
            nn.Linear(in_channels, q_heads * q_channels, bias=False),
            Rearrange("b s (h c) -> b h s c", h=q_heads, c=q_channels),
            nn.LayerNorm(q_channels),
            RoPE(q_channels, seq_len),
        )

        self.k_proj = nn.Sequential(
            nn.Linear(in_channels, kv_heads * k_channels, bias=False),
            Rearrange("b s (h c) -> b h s c", h=kv_heads, c=k_channels),
            nn.LayerNorm(k_channels),
            RoPE(k_channels, seq_len),
        )

        self.v_proj = nn.Sequential(
            nn.Linear(in_channels, kv_heads * v_channels, bias=False),
            Rearrange("b s (h c) -> b h s c", h=kv_heads, c=v_channels),
            nn.LayerNorm(v_channels),
        )

        self.liner1 = nn.Linear(in_channels, in_channels)
        self.norm1 = RMSBatchNorm(in_channels)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x, attn_bias):
        x_norm = self.norm0(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        attn_logits = torch.einsum("b h s c, b 1 S c -> b h s S", q, k) / np.sqrt(
            k.shape[-1]
        )
        attn_logits = torch.tanh((attn_logits + attn_bias) / 5.0) * 5.0
        attn_weights = F.softmax(attn_logits, dim=-1)
        y = torch.einsum("b h s S, b 1 S c -> b h s c", attn_weights, v)
        y = rearrange(y, "b h s c -> b s (h c)")
        y = self.liner1(y)
        y = self.norm1(y)
        return self.dropout1(y)


class Sequence2PairBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        pool_size = 16
        qk_heads = 32
        qk_channels = 128

        self.downsample = nn.Sequential(
            Reduce("b (n p) c -> b p c", "mean", n=pool_size),  # (b, s, c) -> (b, p, c)
            RMSBatchNorm(in_channels),
        )

        self.q_proj = nn.Sequential(
            nn.Linear(in_channels, qk_heads * qk_channels, bias=False),
            Rearrange("b p (h c) -> b p h c", h=qk_heads, c=qk_channels),
        )

        self.k_proj = nn.Sequential(
            nn.Linear(in_channels, qk_heads * qk_channels, bias=False),
            Rearrange("b p (h c) -> b p h c", h=qk_heads, c=qk_channels),
        )

        pair_seq_len = 512
        pair_fea_size = 64

        pos_features = central_mask_features(pair_seq_len, pair_fea_size)
        self.register_buffer("pos_features", pos_features)

        self.to_pos = nn.Linear(pair_fea_size, qk_heads * qk_channels)

        self.register_parameter(
            "q_bias", nn.Parameter(torch.zeros(1, 1, qk_heads, qk_channels))
        )
        self.register_parameter(
            "k_bias", nn.Parameter(torch.zeros(1, 1, qk_heads, qk_channels))
        )

        self.y_q_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_channels, qk_channels, bias=False),
        )

        self.y_k_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(in_channels, qk_channels, bias=False),
        )

        self.pair_proj = nn.Linear(qk_heads, qk_channels)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.downsample(x)
        q = self.q_proj(x)
        k = self.k_proj(x)

        pos_encodings = self.to_pos(self.pos_features)

        rel_q_a = relative_shift(
            torch.einsum("b p h c, q h c -> b h p q", q + self.q_bias, pos_encodings)
        )
        rel_q_a = rearrange(rel_q_a, "b h p P -> b p P h")

        rel_k_a = relative_shift(
            torch.einsum("b p h c, q h c -> b h p q", k + self.k_bias, pos_encodings)
        )
        rel_k_a = rearrange(rel_k_a, "b h p P -> b P p h")

        a = torch.einsum("b p h c, b P h c -> b p P h", q, k) + (rel_q_a + rel_k_a) / 2

        y_q = self.y_q_proj(x)
        y_k = self.y_k_proj(x)

        pair_actvations = self.pair_proj(a) + y_q[:, :, None, :] + y_k[:, None, :, :]

        return self.dropout(pair_actvations)


def central_mask_features(sequence_length: int, feature_size: int):
    relative_positions = torch.arange(2 * sequence_length - 1) - (sequence_length - 1)
    center_widths = torch.arange(feature_size // 2) + np.geomspace(
        1, sequence_length - feature_size // 2 + 1, feature_size // 2, endpoint=False
    )
    embeddings = center_widths[None, :] > torch.abs(relative_positions)[:, None]
    return torch.cat(
        [embeddings, torch.sign(relative_positions)[:, None] * embeddings], axis=-1
    )


# data shape: (...,  S,  2*S-1) -> (..., S, S)
def relative_shift(x):
    *batch_shapes, seq_length, num_diagonals = x.shape
    x = F.pad(x, (1, 0))
    x = x.reshape(batch_shapes + [num_diagonals + 1, seq_length])
    x = x[..., 1:, :].reshape(batch_shapes + [seq_length, num_diagonals])
    return x[..., :seq_length]
