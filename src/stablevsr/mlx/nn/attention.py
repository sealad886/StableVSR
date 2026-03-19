"""Attention and cross-attention layers for MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.core.fast as fast  # type: ignore[import-not-found]
import mlx.nn as nn


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional encoder hidden states."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim or query_dim

        self.heads = heads
        self.dim_head = dim_head
        self.scale = 1.0 / math.sqrt(dim_head)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = [nn.Linear(inner_dim, query_dim)]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch, seq_len, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # Reshape to (B, heads, seq, dim_head)
        q = q.reshape(batch, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(batch, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(batch, -1, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        out = fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        return self.to_out[0](out)


class GEGLU(nn.Module):
    """GELU-gated linear unit used in transformer feed-forward."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return x * nn.gelu_approx(gate)


class FeedForward(nn.Module):
    """Transformer feed-forward with GEGLU activation.

    Matches diffusers indexing: net.0=GEGLU, net.1=Identity(dropout placeholder), net.2=Linear.
    """

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net = [GEGLU(dim, inner_dim), nn.Identity(), nn.Linear(inner_dim, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.net:
            x = layer(x)
        return x


class BasicTransformerBlock(nn.Module):
    """Single transformer block: self-attn → cross-attn → FF."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int | None = None,
        only_cross_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        # Self-attention (or cross-attention if only_cross_attention)
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn1(
            hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
        )
        hidden_states = hidden_states + residual

        # Cross-attention
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn2(
            hidden_states, encoder_hidden_states=encoder_hidden_states
        )
        hidden_states = hidden_states + residual

        # Feed-forward
        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class Transformer2DModel(nn.Module):
    """Spatial transformer: project spatial→seq, apply transformer blocks, project back."""

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_layers: int = 1,
        cross_attention_dim: int | None = None,
        only_cross_attention: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = nn.GroupNorm(32, in_channels, pytorch_compatible=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = [
            BasicTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                cross_attention_dim=cross_attention_dim,
                only_cross_attention=only_cross_attention,
            )
            for _ in range(num_layers)
        ]

        self.proj_out = nn.Linear(inner_dim, in_channels)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        # hidden_states: (B, H, W, C)  [NHWC]
        batch, height, width, _ = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.reshape(batch, height * width, -1)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states=encoder_hidden_states
            )

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, width, -1)
        hidden_states = hidden_states + residual

        return hidden_states
