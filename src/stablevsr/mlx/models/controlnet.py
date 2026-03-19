"""ControlNet for MLX.

Config: in_channels=7, conditioning_channels=3, cross_attention_dim=1024,
        block_out_channels=[256,512,512,1024],
        only_cross_attention=[true,true,true,false]

The ControlNet mirrors the UNet encoder + mid block but adds:
- A separate conditioning input path (3ch → hint processing)
- Zero-conv output projections for each down block + mid block
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .unet import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    TimestepEmbedding,
    UNetMidBlock2DCrossAttn,
)


class ControlNetConditioningEmbedding(nn.Module):
    """Process the conditioning image (warped previous frame estimate)."""

    def __init__(
        self,
        conditioning_channels: int = 3,
        block_out_channels: tuple[int, ...] = (16, 32, 96, 256),
        conditioning_embedding_out_channels: int = 256,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], 3, padding=1)

        self.blocks = []
        ch_in = block_out_channels[0]
        for ch_out in block_out_channels[1:]:
            self.blocks.append(nn.Conv2d(ch_in, ch_in, 3, padding=1))
            self.blocks.append(nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1))
            ch_in = ch_out

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_out_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.silu(self.conv_in(x))
        for block in self.blocks:
            x = nn.silu(block(x))
        x = self.conv_out(x)
        return x


def _zero_conv(channels: int) -> nn.Conv2d:
    """Create a zero-initialized 1x1 conv for ControlNet outputs."""
    conv = nn.Conv2d(channels, channels, 1)
    conv.weight = mx.zeros_like(conv.weight)
    conv.bias = mx.zeros_like(conv.bias)
    return conv


class ControlNetModel(nn.Module):
    """ControlNet that mirrors UNet encoder to inject temporal conditioning."""

    def __init__(
        self,
        in_channels: int = 7,
        conditioning_channels: int = 3,
        block_out_channels: tuple[int, ...] = (256, 512, 512, 1024),
        layers_per_block: int = 2,
        cross_attention_dim: int = 1024,
        attention_head_dim: int = 8,
        only_cross_attention: tuple[bool, ...] = (True, True, True, False),
        conditioning_embedding_out_channels: tuple[int, ...] = (64, 128, 256),
        down_block_types: tuple[str, ...] = (
            "DownBlock2D", "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
    ):
        super().__init__()
        time_embed_dim = block_out_channels[0] * 4

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        # Conditioning embedding for the control image
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_channels=conditioning_channels,
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_embedding_out_channels=block_out_channels[0],
        )

        # Zero convs for outputs
        self.controlnet_down_blocks = [_zero_conv(block_out_channels[0])]

        n_heads = [ch // attention_head_dim for ch in block_out_channels]

        # Down blocks — follow config order exactly
        self.down_blocks: list = []
        ch_in = block_out_channels[0]
        for i, (ch_out, btype) in enumerate(zip(block_out_channels, down_block_types)):
            is_last = i == len(block_out_channels) - 1
            if btype == "CrossAttnDownBlock2D":
                self.down_blocks.append(
                    CrossAttnDownBlock2D(
                        ch_in, ch_out, time_embed_dim,
                        num_layers=layers_per_block,
                        num_attention_heads=n_heads[i],
                        cross_attention_dim=cross_attention_dim,
                        add_downsample=not is_last,
                        only_cross_attention=only_cross_attention[i],
                    )
                )
            else:
                self.down_blocks.append(
                    DownBlock2D(
                        ch_in, ch_out, time_embed_dim,
                        num_layers=layers_per_block,
                        add_downsample=not is_last,
                    )
                )

            for _ in range(layers_per_block):
                self.controlnet_down_blocks.append(_zero_conv(ch_out))
            if not is_last:
                self.controlnet_down_blocks.append(_zero_conv(ch_out))

            ch_in = ch_out

        # Mid block
        self.mid_block = UNetMidBlock2DCrossAttn(
            block_out_channels[-1], time_embed_dim,
            num_attention_heads=n_heads[-1],
            cross_attention_dim=cross_attention_dim,
        )
        self.controlnet_mid_block = _zero_conv(block_out_channels[-1])

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        controlnet_cond: mx.array,
        conditioning_scale: float = 1.0,
    ) -> tuple[list[mx.array], mx.array]:
        """Forward pass.

        Returns:
            (down_block_res_samples, mid_block_res_sample) scaled by conditioning_scale.
        """
        # Time embedding
        temb = self.time_embedding(timestep)

        # Process input sample
        sample = self.conv_in(sample)

        # Add conditioning embedding
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # Down blocks
        down_block_res_samples = [self.controlnet_down_blocks[0](sample)]
        ctrl_idx = 1

        for block in self.down_blocks:
            sample, res_samples = block(sample, temb, encoder_hidden_states)
            for res in res_samples:
                down_block_res_samples.append(self.controlnet_down_blocks[ctrl_idx](res))
                ctrl_idx += 1

        # Mid block
        sample = self.mid_block(sample, temb, encoder_hidden_states)
        mid_block_res_sample = self.controlnet_mid_block(sample)

        # Apply conditioning scale
        down_block_res_samples = [s * conditioning_scale for s in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        return down_block_res_samples, mid_block_res_sample
