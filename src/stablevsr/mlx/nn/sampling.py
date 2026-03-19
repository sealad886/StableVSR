"""Upsample and Downsample blocks for MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class Upsample2D(nn.Module):
    """2x spatial upsample via nearest-neighbor + Conv2d."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1) if use_conv else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) NHWC
        B, H, W, C = x.shape
        x = mx.repeat(x, repeats=2, axis=1)
        x = mx.repeat(x, repeats=2, axis=2)
        if self.conv is not None:
            x = self.conv(x)
        return x


class Downsample2D(nn.Module):
    """2x spatial downsample via stride-2 Conv2d."""

    def __init__(self, channels: int, use_conv: bool = True, padding: int = 1):
        super().__init__()
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=padding)
        else:
            self.conv = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.conv is not None:
            return self.conv(x)
        # Average pooling fallback
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C)
        return mx.mean(x, axis=(2, 4))
