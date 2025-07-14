import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from einops.layers.torch import Rearrange, Reduce
from einops import repeat, rearrange


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
        fan_in = weight.shape[1:].numel()  # in_channels * kernel_size
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


class cbConvBlock(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = Conv1D(in_channels, out_channels, 15, padding=15 // 2)
        self.conv1 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.conv0(x)
        return out + self.conv1(out)


class DownresBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pad = out_channels - in_channels
        self.conv0 = ConvBlock(in_channels, out_channels)
        self.conv1 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.conv0(x)
        out = out + F.pad(x, (0, self.pad))
        return out + self.conv1(out)


class SequenceEncoder(nn.Module):
    def __init__(self, base_channels, down_channels, n_down_blocks):
        super().__init__()
        embedding_channels = down_channels[0]
        self.down_blocks = nn.ModuleList(
            [DnaEmbedder(base_channels, embedding_channels)]
            + [DownresBlock(down_channels[i], down_channels[i+1]) for i in range(n_down_blocks)]
        )

    def forward(self, x):
        intermediates = []

        for block in self.down_blocks:
            x = block(x)
            intermediates.append(x)
            # Maxpool
        return x, intermediates


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


class RowAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.RMSNorm(in_channels)
        qkv_channels = 128
        self.q_proj = nn.Linear(in_channels, qkv_channels, bias=False)
        self.k_proj = nn.Linear(in_channels, qkv_channels, bias=False)
        self.v_proj = nn.Linear(in_channels, qkv_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x_norm = self.norm(x)
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        attn_weights = torch.einsum("b p P f, b p k f -> b p P k", q, k) / np.sqrt(
            k.shape[-1]
        )
        attn_weights = F.softmax(attn_weights, dim=-1)

        y = torch.einsum("b p P k, b p k f -> b p P f", attn_weights, v)
        return self.dropout(y)


class PairMlpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels * 2
        dropout_rate = 0.3
        self.net = nn.Sequential(
            nn.RMSNorm(in_channels),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class PairUpdateBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        pair_channels = 128
        self.seq2pair = Sequence2PairBlock(in_channels)
        self.row_attn = RowAttentionBlock(pair_channels)
        self.pair_mlp = PairMlpBlock(pair_channels)

    def forward(self, seq_input, pair_input):
        y = self.seq2pair(seq_input)
        x = y if pair_input is None else pair_input + y
        x += self.row_attn(x)
        x += self.pair_mlp(x)
        return x


class TransformerTower(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        layer_depth = 9
        pairwise_interval = 2
        seq_len = 8192
        pair_channels = 128
        layers = []
        for i in range(layer_depth):
            if i % pairwise_interval == 0:
                pair_update = PairUpdateBlock(in_channels)
            else:
                pair_update = None

            mha = MhaBlock(in_channels, seq_len)
            attn_bias = AttentionBiasBlock(pair_channels)
            mlp = MlpBlock(in_channels)

            layers.append(nn.ModuleList([pair_update, mha, attn_bias, mlp]))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        pair_x = None
        for pair_update, mha, attn_bias, mlp in self.layers:
            if pair_update is not None:
                pair_x = pair_update(x, pair_x)

            x = x + mha(x, attn_bias(pair_x))
            x = x + mlp(x)

        return x, pair_x


class UpresBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pad = in_channels - out_channels
        self.conv0 = ConvBlock(in_channels, out_channels)
        nn.register_parameter(self, "residual_scale", torch.tensor(0.9))
        self.conv_unet = ConvBlock(out_channels, out_channels, 1)
        self.conv1 = ConvBlock(out_channels, out_channels)

    def forward(self, x, unet_skip):
        out = self.conv0(x) + x[..., : -self.pad]
        out = repeat(out, "b s c -> b (s 2) c") * self.residual_scale
        out += self.conv_unet(unet_skip)
        return out + self.conv1(out)


class SequenceDecoder(nn.Module):
    def __init__(self, up_channels, n_up_blocks):
        super().__init__()
        self.up_blocks = nn.ModuleList(
            [UpresBlock(up_channels[i], up_channels[i+1]) for i in range(n_up_blocks)]
        )

    def forward(self, x, intermediates):
        for block in self.up_blocks:
            x = block(x, intermediates.pop())

        return x
    

class TransformerUnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SequenceEncoder(config.base_channels, config.down_channels, config.n_down_blocks)
        self.transformer = TransformerTower(config.transformer_channels)
        self.decoder = SequenceDecoder(config.up_channels, config.n_up_blocks)

    def forward(self, x):
        x, intermediates = self.encoder(x)
        x, pair_x = self.transformer(x)
        x = self.decoder(x, intermediates[::-1])  # Reverse the order of intermediates
        return x, pair_x