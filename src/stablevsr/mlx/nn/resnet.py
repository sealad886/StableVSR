"""ResNet blocks for MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class ResnetBlock2D(nn.Module):
    """ResNet block with optional shortcut projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        temb_channels: int = 512,
        groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Timestep embedding projection
        if temb_channels > 0:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def __call__(self, x: mx.array, temb: mx.array | None = None) -> mx.array:
        residual = x

        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)

        if temb is not None and self.time_emb_proj is not None:
            temb = nn.silu(temb)
            temb = self.time_emb_proj(temb)
            # temb: (B, C) → (B, 1, 1, C) for NHWC
            x = x + temb[:, None, None, :]

        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return x + residual
