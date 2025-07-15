from dataclasses import dataclass, field
from typing import List


@dataclass
class AGConfig:
    """
    Configuration for the AlphaGenome model.
    """

    # Unet parameters
    base_channels: int = 4
    embedding_channels: int = 768
    channel_step: int = 128

    n_down_blocks: int = 6
    n_up_blocks: int = 7

    # Transformer parameters
    n_transformer_blocks: int = 9
    q_heads: int = 8
    qk_channels: int = 128
    v_channels: int = 192
    max_position: int = 8192
    single_seq_len: int = 8192

    # pairwise attention
    pair_block_gap: int = 2
    pair_pool_size: int = 16
    pair_qk_heads: int = 32
    pair_qk_channels: int = 128
    pair_pos_feat_size: int = 64

    # Auto generated parameters
    transformer_channels: int = field(default_factory=int, init=False)
    down_channels: List[int] = field(default_factory=list, init=False)
    up_channels: List[int] = field(default_factory=list, init=False)
    pair_seq_len: int = field(default_factory=int, init=False)

    def __post_init__(self):
        self.transformer_channels = (
            self.embedding_channels + self.channel_step * self.n_down_blocks
        )

        self.down_channels = [
            self.embedding_channels + i * self.channel_step
            for i in range(self.n_down_blocks)
        ] + [self.transformer_channels]

        self.up_channels = [self.transformer_channels] + self.down_channels[::-1]

        self.pair_seq_len = self.single_seq_len // self.pair_pool_size
