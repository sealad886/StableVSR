"""CLIP text encoder for MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.core.fast as fast  # type: ignore[import-not-found]
import mlx.nn as nn


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        seq_len = input_ids.shape[1]
        position_ids = mx.arange(seq_len)
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(
        self, hidden_states: mx.array, causal_mask: mx.array | None = None
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(hidden_states)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(hidden_states)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        out = fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=causal_mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_fast_approx(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def __call__(
        self, hidden_states: mx.array, causal_mask: mx.array | None = None
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int, num_layers: int
    ):
        super().__init__()
        self.layers = [
            CLIPEncoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ]

    def __call__(
        self, hidden_states: mx.array, causal_mask: mx.array | None = None
    ) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)
        return hidden_states


class CLIPTextModel(nn.Module):
    """CLIP text encoder: embeddings → transformer → final layer norm.

    Config: hidden_size=1024, num_heads=16, num_layers=23,
            intermediate_size=4096, vocab_size=49408, max_position_embeddings=77
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 23,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 77,
    ):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(
            vocab_size, hidden_size, max_position_embeddings
        )
        self.encoder = CLIPEncoder(
            hidden_size, num_attention_heads, intermediate_size, num_hidden_layers
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        self._max_position_embeddings = max_position_embeddings

    def __call__(self, input_ids: mx.array) -> mx.array:
        """Returns last hidden state (B, seq_len, hidden_size)."""
        hidden = self.embeddings(input_ids)

        # Causal mask
        L = input_ids.shape[1]
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(
            hidden.dtype
        )

        hidden = self.encoder(hidden, causal_mask)
        hidden = self.final_layer_norm(hidden)
        return hidden
