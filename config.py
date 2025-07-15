from dataclasses import dataclass, field
from typing import List


@dataclass
class AGConfig:
    """
    Configuration for the AlphaGenome model.
    """

    # Model parameters
    base_channels = 4
    embedding_channels: int = 768
    channel_step = 128

    n_down_blocks: int = 6
    n_up_blocks: int = 7



    # Auto generated parameters
    transformer_channels: int = field(default_factory=int, init=False)
    down_channels: List[int] = field(default_factory=list, init=False)
    up_channels: List[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.transformer_channels = (
            self.embedding_channels + self.channel_step * self.n_down_blocks
        )

        self.down_channels = [
            self.embedding_channels + i * self.channel_step
            for i in range(self.n_down_blocks)
        ] + [self.transformer_channels]

        self.up_channels = [self.transformer_channels] + self.down_channels[::-1]
