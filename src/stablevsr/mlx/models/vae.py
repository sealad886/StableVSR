"""VAE (AutoencoderKL) for MLX.

Config: latent_channels=4, block_out_channels=[128,256,512],
        layers_per_block=2, scaling_factor=0.08333
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.core.fast as fast  # type: ignore[import-not-found]
import mlx.nn as nn

from ..nn.resnet import ResnetBlock2D
from ..nn.sampling import Downsample2D, Upsample2D


class AttentionBlock(nn.Module):
    """Simple attention block used in VAE (not the full Transformer2D).

    Matches diffusers naming: group_norm, to_q, to_k, to_v, to_out.0
    """

    def __init__(self, channels: int, num_head_channels: int = 1):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

        self.channels = channels
        self.num_heads = channels // num_head_channels if num_head_channels > 1 else 1
        self.head_dim = channels // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) NHWC
        batch, height, width, channels = x.shape
        residual = x

        x = self.group_norm(x)
        x = x.reshape(batch, height * width, channels)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape for multi-head attention
        q = q.reshape(batch, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        out = fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(batch, height * width, channels)

        out = self.to_out[0](out)
        out = out.reshape(batch, height, width, channels)

        return out + residual


class VAEMidBlock(nn.Module):
    """Mid block for VAE encoder/decoder. Contains resnets + simple attention."""

    def __init__(self, channels: int, norm_num_groups: int = 32):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(channels, channels, temb_channels=0, groups=norm_num_groups),
            ResnetBlock2D(channels, channels, temb_channels=0, groups=norm_num_groups),
        ]
        self.attentions = [AttentionBlock(channels)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)
        return x


class VAEEncoder(nn.Module):
    """Encoder: (B,H,W,3) → (B,H/8,W/8,latent_channels*2) for mean+logvar."""

    def __init__(
        self,
        in_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512),
        layers_per_block: int = 2,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)

        self.down_blocks = []
        ch_in = block_out_channels[0]
        for i, ch_out in enumerate(block_out_channels):
            resnets = [
                ResnetBlock2D(ch_in if j == 0 else ch_out, ch_out, temb_channels=0, groups=norm_num_groups)
                for j in range(layers_per_block)
            ]
            downsample = Downsample2D(ch_out) if i < len(block_out_channels) - 1 else None
            self.down_blocks.append({"resnets": resnets, "downsamplers": [downsample] if downsample else []})
            ch_in = ch_out

        # Mid block
        self.mid_block = VAEMidBlock(ch_in, norm_num_groups)

        self.conv_norm_out = nn.GroupNorm(norm_num_groups, ch_in, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(ch_in, 2 * latent_channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)

        for block in self.down_blocks:
            for resnet in block["resnets"]:
                x = resnet(x)
            if block["downsamplers"]:
                x = block["downsamplers"][0](x)

        x = self.mid_block(x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x


class VAEDecoder(nn.Module):
    """Decoder: (B,H/8,W/8,latent) → (B,H,W,3)."""

    def __init__(
        self,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512),
        layers_per_block: int = 2,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))
        ch_in = reversed_channels[0]

        self.conv_in = nn.Conv2d(latent_channels, ch_in, 3, padding=1)

        # Mid block
        self.mid_block = VAEMidBlock(ch_in, norm_num_groups)

        self.up_blocks = []
        for i, ch_out in enumerate(reversed_channels):
            resnets = [
                ResnetBlock2D(ch_in if j == 0 else ch_out, ch_out, temb_channels=0, groups=norm_num_groups)
                for j in range(layers_per_block + 1)
            ]
            upsample = Upsample2D(ch_out) if i < len(reversed_channels) - 1 else None
            self.up_blocks.append({"resnets": resnets, "upsamplers": [upsample] if upsample else []})
            ch_in = ch_out

        self.conv_norm_out = nn.GroupNorm(norm_num_groups, reversed_channels[-1], pytorch_compatible=True)
        self.conv_out = nn.Conv2d(reversed_channels[-1], out_channels, 3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)

        x = self.mid_block(x)

        for block in self.up_blocks:
            for resnet in block["resnets"]:
                x = resnet(x)
            if block["upsamplers"]:
                x = block["upsamplers"][0](x)

        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x


class AutoencoderKL(nn.Module):
    """VAE with KL divergence for latent diffusion.

    Encode: image → latent (sample from distribution)
    Decode: latent → image
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512),
        layers_per_block: int = 2,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.08333,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.encoder = VAEEncoder(
            in_channels, block_out_channels, layers_per_block, latent_channels, norm_num_groups
        )
        self.decoder = VAEDecoder(
            out_channels, block_out_channels, layers_per_block, latent_channels, norm_num_groups
        )
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def encode(self, x: mx.array) -> mx.array:
        """Encode to latent distribution parameters."""
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = mx.split(h, 2, axis=-1)
        return mean

    def decode(self, z: mx.array) -> mx.array:
        """Decode latent to image."""
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def tiled_decode(
        self,
        z: mx.array,
        tile_size: int = 64,
        tile_overlap: int = 16,
    ) -> mx.array:
        """Decode latent in tiles to avoid OOM from VAE mid-block attention.

        For this VAE with block_out_channels=[128,256,512] (vae_scale_factor=4),
        the bottleneck attention operates at full latent spatial resolution. Tiling
        keeps each attention to tile_size² tokens instead of (H*W)² tokens.
        """
        batch, h_lat, w_lat, c = z.shape
        sf = 2 ** (len(self.decoder.up_blocks) - 1)
        stride = tile_size - tile_overlap

        out_h = h_lat * sf
        out_w = w_lat * sf
        out_tile = tile_size * sf
        out_overlap = tile_overlap * sf

        # 1-D linear blend ramp for overlap blending
        ramp = mx.concatenate([
            mx.linspace(0.0, 1.0, out_overlap),
            mx.ones(out_tile - 2 * out_overlap),
            mx.linspace(1.0, 0.0, out_overlap),
        ])
        # 2-D separable blend weight (H_tile, W_tile, 1)
        blend_h = ramp[: out_tile, None, None]
        blend_w = ramp[None, : out_tile, None]

        output = mx.zeros((batch, out_h, out_w, 3))
        weight = mx.zeros((1, out_h, out_w, 1))

        for y in range(0, h_lat, stride):
            for x in range(0, w_lat, stride):
                # Clamp tile boundaries
                y_end = min(y + tile_size, h_lat)
                x_end = min(x + tile_size, w_lat)
                y_start = max(y_end - tile_size, 0)
                x_start = max(x_end - tile_size, 0)

                tile = z[:, y_start:y_end, x_start:x_end, :]
                decoded = self.decode(tile)
                mx.eval(decoded)

                oh, ow = decoded.shape[1], decoded.shape[2]
                oy, ox = y_start * sf, x_start * sf

                # Build per-tile blend mask (handle edge tiles smaller than full)
                bh = blend_h[:oh]
                bw = blend_w[:, :ow]
                tile_blend = bh * bw  # (oh, ow, 1)

                output[:, oy : oy + oh, ox : ox + ow, :] += decoded * tile_blend
                weight[:, oy : oy + oh, ox : ox + ow, :] += tile_blend
                mx.eval(output, weight)

        return output / mx.maximum(weight, 1e-8)
