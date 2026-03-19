"""UNet2D Condition Model for MLX.

Config: in_channels=7, out_channels=4, cross_attention_dim=1024,
        block_out_channels=[256,512,512,1024],
        down_block_types=["CrossAttnDownBlock2D"x3, "DownBlock2D"],
        up_block_types=["CrossAttnUpBlock2D"x3, "UpBlock2D"],
        layers_per_block=2, attention_head_dim=8
"""

from __future__ import annotations

import logging
import math

import mlx.core as mx
import mlx.nn as nn

from ..nn.attention import Transformer2DModel
from ..nn.resnet import ResnetBlock2D
from ..nn.sampling import Downsample2D, Upsample2D

logger = logging.getLogger(__name__)


def match_spatial_dims(
    hidden_states: mx.array,
    target: mx.array,
) -> mx.array:
    """Crop hidden_states to match target's spatial dimensions if needed.

    After nearest-neighbor 2× upsampling, the upsampled tensor can be exactly
    1 pixel larger than the corresponding skip connection when the skip was
    produced from an odd-sized input (e.g. 135 → floor(135/2)=67 → 67*2=134,
    but skip is 135). This is an inherent property of integer division in
    downsample/upsample pairs.

    Invariants:
        - hidden_states.shape[1] >= target.shape[1]  (height)
        - hidden_states.shape[2] >= target.shape[2]  (width)
        - Difference is at most 1 pixel per axis
        - Crop is always from the bottom/right edge (top-left anchored)
        - When shapes already match, this is a no-op (returns input unchanged)

    This matches the behavior of diffusers' PyTorch implementation, where
    F.interpolate produces exact target sizes via output_size parameter.
    """
    h_diff = hidden_states.shape[1] - target.shape[1]
    w_diff = hidden_states.shape[2] - target.shape[2]
    if h_diff == 0 and w_diff == 0:
        return hidden_states
    if h_diff < 0 or w_diff < 0:
        raise ValueError(
            f"Upsampled tensor ({hidden_states.shape}) is smaller than skip "
            f"connection ({target.shape}). This indicates a structural bug in the "
            f"UNet architecture, not an off-by-one from upsampling."
        )
    if h_diff > 1 or w_diff > 1:
        raise ValueError(
            f"Spatial mismatch of {h_diff}×{w_diff} exceeds expected ±1. "
            f"hidden_states={hidden_states.shape}, target={target.shape}. "
            f"This may indicate mismatched encoder/decoder block counts."
        )
    logger.debug(
        "Cropping hidden_states %s → (%d, %d) to match skip connection",
        hidden_states.shape,
        target.shape[1],
        target.shape[2],
    )
    return hidden_states[:, : target.shape[1], : target.shape[2], :]


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep → linear projection."""

    def __init__(self, dim: int, time_embed_dim: int):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, t: mx.array) -> mx.array:
        # Sinusoidal embedding
        half_dim = self.dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half_dim) / half_dim)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if self.dim % 2:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1))], axis=-1)

        emb = nn.silu(self.linear_1(emb))
        emb = self.linear_2(emb)
        return emb


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 1024,
        add_downsample: bool = True,
        only_cross_attention: bool = False,
    ):
        super().__init__()
        attn_head_dim = out_channels // num_attention_heads

        self.resnets = [
            ResnetBlock2D(
                in_channels if i == 0 else out_channels, out_channels, temb_channels
            )
            for i in range(num_layers)
        ]
        self.attentions = [
            Transformer2DModel(
                num_attention_heads,
                attn_head_dim,
                out_channels,
                num_layers=1,
                cross_attention_dim=cross_attention_dim,
                only_cross_attention=only_cross_attention,
            )
            for _ in range(num_layers)
        ]
        self.downsamplers = [Downsample2D(out_channels)] if add_downsample else []

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            output_states.append(hidden_states)

        if self.downsamplers:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        add_downsample: bool = True,
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(
                in_channels if i == 0 else out_channels, out_channels, temb_channels
            )
            for i in range(num_layers)
        ]
        self.downsamplers = [Downsample2D(out_channels)] if add_downsample else []

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> tuple[mx.array, list[mx.array]]:
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsamplers:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 1024,
        add_upsample: bool = True,
        only_cross_attention: bool = False,
    ):
        super().__init__()
        attn_head_dim = out_channels // num_attention_heads

        self.resnets = [
            ResnetBlock2D(
                (prev_output_channel if i == 0 else out_channels)
                + (out_channels if i > 0 else in_channels),
                out_channels,
                temb_channels,
            )
            for i in range(num_layers + 1)
        ]
        self.attentions = [
            Transformer2DModel(
                num_attention_heads,
                attn_head_dim,
                out_channels,
                num_layers=1,
                cross_attention_dim=cross_attention_dim,
                only_cross_attention=only_cross_attention,
            )
            for _ in range(num_layers + 1)
        ]
        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else []

    def __call__(
        self,
        hidden_states: mx.array,
        res_hidden_states_tuple: list[mx.array],
        temb: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        for resnet, attn in zip(self.resnets, self.attentions):
            res = res_hidden_states_tuple.pop()
            hidden_states = match_spatial_dims(hidden_states, res)
            hidden_states = mx.concatenate([hidden_states, res], axis=-1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)

        if self.upsamplers:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_layers: int = 2,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(
                (prev_output_channel if i == 0 else out_channels)
                + (out_channels if i > 0 else in_channels),
                out_channels,
                temb_channels,
            )
            for i in range(num_layers + 1)
        ]
        self.upsamplers = [Upsample2D(out_channels)] if add_upsample else []

    def __call__(
        self,
        hidden_states: mx.array,
        res_hidden_states_tuple: list[mx.array],
        temb: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        for resnet in self.resnets:
            res = res_hidden_states_tuple.pop()
            hidden_states = match_spatial_dims(hidden_states, res)
            hidden_states = mx.concatenate([hidden_states, res], axis=-1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers:
            hidden_states = self.upsamplers[0](hidden_states)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 1024,
    ):
        super().__init__()
        attn_head_dim = in_channels // num_attention_heads

        self.resnets = [
            ResnetBlock2D(in_channels, in_channels, temb_channels),
            ResnetBlock2D(in_channels, in_channels, temb_channels),
        ]
        self.attentions = [
            Transformer2DModel(
                num_attention_heads,
                attn_head_dim,
                in_channels,
                num_layers=1,
                cross_attention_dim=cross_attention_dim,
            ),
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class UNet2DConditionModel(nn.Module):
    """UNet with cross-attention conditioning and ControlNet residual injection.

    in_channels=7 (4 latent + 3 image concat).
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 4,
        block_out_channels: tuple[int, ...] = (256, 512, 512, 1024),
        layers_per_block: int = 2,
        cross_attention_dim: int = 1024,
        attention_head_dim: int = 8,
        only_cross_attention: tuple[bool, ...] | bool = False,
        down_block_types: tuple[str, ...] = (
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types: tuple[str, ...] = (
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
    ):
        super().__init__()
        time_embed_dim = block_out_channels[0] * 4

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        if isinstance(only_cross_attention, bool):
            only_cross_attention = (only_cross_attention,) * len(block_out_channels)

        # Down blocks — follow config order exactly
        self.down_blocks: list = []
        ch_in = block_out_channels[0]
        n_heads = [ch // attention_head_dim for ch in block_out_channels]

        for i, (ch_out, btype) in enumerate(zip(block_out_channels, down_block_types)):
            is_last = i == len(block_out_channels) - 1
            if btype == "CrossAttnDownBlock2D":
                self.down_blocks.append(
                    CrossAttnDownBlock2D(
                        ch_in,
                        ch_out,
                        time_embed_dim,
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
                        ch_in,
                        ch_out,
                        time_embed_dim,
                        num_layers=layers_per_block,
                        add_downsample=not is_last,
                    )
                )
            ch_in = ch_out

        # Mid block
        self.mid_block = UNetMidBlock2DCrossAttn(
            block_out_channels[-1],
            time_embed_dim,
            num_attention_heads=n_heads[-1],
            cross_attention_dim=cross_attention_dim,
        )

        # Up blocks — follow config order exactly
        self.up_blocks: list = []
        reversed_channels = list(reversed(block_out_channels))
        reversed_only_cross = list(reversed(only_cross_attention))

        ch_in = reversed_channels[0]
        for i, (ch_out, btype) in enumerate(zip(reversed_channels, up_block_types)):
            is_last = i == len(reversed_channels) - 1
            prev_ch = reversed_channels[max(i - 1, 0)]
            if btype == "CrossAttnUpBlock2D":
                self.up_blocks.append(
                    CrossAttnUpBlock2D(
                        ch_in,
                        ch_out,
                        prev_ch,
                        time_embed_dim,
                        num_layers=layers_per_block,
                        num_attention_heads=n_heads[len(block_out_channels) - 1 - i],
                        cross_attention_dim=cross_attention_dim,
                        add_upsample=not is_last,
                        only_cross_attention=reversed_only_cross[i],
                    )
                )
            else:
                self.up_blocks.append(
                    UpBlock2D(
                        ch_in,
                        ch_out,
                        prev_ch,
                        time_embed_dim,
                        num_layers=layers_per_block,
                        add_upsample=not is_last,
                    )
                )
            ch_in = ch_out

        self.conv_norm_out = nn.GroupNorm(
            32, block_out_channels[0], pytorch_compatible=True
        )
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def __call__(
        self,
        sample: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        down_block_additional_residuals: list[mx.array] | None = None,
        mid_block_additional_residual: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            sample: (B, H, W, 7) NHWC — latent concat with image.
            timestep: (B,) integer timesteps.
            encoder_hidden_states: (B, seq, 1024) text embeddings.
            down_block_additional_residuals: ControlNet residuals for down blocks.
            mid_block_additional_residual: ControlNet residual for mid block.

        Returns:
            Noise prediction (B, H, W, 4) NHWC.
        """
        # 1. Time embedding
        temb = self.time_embedding(timestep)

        # 2. Pre-process
        sample = self.conv_in(sample)

        # 3. Down
        down_block_res_samples = [sample]
        for block in self.down_blocks:
            sample, res_samples = block(sample, temb, encoder_hidden_states)
            down_block_res_samples.extend(res_samples)

        # 4. ControlNet residuals for down blocks
        if down_block_additional_residuals is not None:
            new_res = []
            for res, ctrl in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                new_res.append(res + ctrl)
            down_block_res_samples = new_res

        # 5. Mid
        sample = self.mid_block(sample, temb, encoder_hidden_states)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 6. Up
        for block in self.up_blocks:
            # Take the correct number of residuals from the stack
            n_resnets = len(block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]
            sample = block(sample, res_samples, temb, encoder_hidden_states)

        # 7. Post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)

        return sample
