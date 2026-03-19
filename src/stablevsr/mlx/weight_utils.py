"""Weight loading utilities: safetensors → MLX with NCHW→NHWC conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx


def _conv_transpose_keys(config: dict[str, Any]) -> set[str]:
    """Build set of weight-key prefixes that correspond to ConvTranspose2d layers.

    Uses the model config (block types containing 'Up') to identify transpose
    conv layers rather than fragile string heuristics.
    """
    prefixes: set[str] = set()
    up_block_types = config.get("up_block_types", [])
    for i, _ in enumerate(up_block_types):
        prefixes.add(f"up_blocks.{i}.upsamplers.0.conv")
    if config.get("_class_name") == "AutoencoderKL":
        for i, _ in enumerate(config.get("up_block_types", [])):
            prefixes.add(f"decoder.up_blocks.{i}.upsamplers.0.conv")
    return prefixes


def load_safetensors_for_mlx(
    safetensors_path: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, mx.array]:
    """Load safetensors weights, converting NCHW→NHWC for conv layers.

    Uses config.json (when provided) to precisely identify ConvTranspose2d
    layers for correct weight transposition.
    """
    weights = mx.load(str(safetensors_path))

    transpose_prefixes: set[str] = set()
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            transpose_prefixes = _conv_transpose_keys(config)

    converted: dict[str, mx.array] = {}
    for key, tensor in weights.items():
        if tensor.ndim == 4:
            is_transpose = any(
                key.startswith(p) or f".{p}." in key for p in transpose_prefixes
            )
            if is_transpose:
                # PyTorch ConvTranspose2d: (C_in, C_out, kH, kW) → MLX: (C_out, kH, kW, C_in)
                tensor = mx.transpose(tensor, axes=(1, 2, 3, 0))
            else:
                # PyTorch Conv2d: (C_out, C_in, kH, kW) → MLX: (C_out, kH, kW, C_in)
                tensor = mx.transpose(tensor, axes=(0, 2, 3, 1))
        converted[key] = tensor

    return converted


def validate_shapes(
    loaded: dict[str, mx.array],
    model_params: dict[str, mx.array],
) -> list[str]:
    """Validate loaded weight shapes against model parameters. Returns mismatches."""
    errors = []
    for key, param in model_params.items():
        if key not in loaded:
            errors.append(f"Missing: {key}")
        elif loaded[key].shape != param.shape:
            errors.append(
                f"Shape mismatch: {key}: loaded={loaded[key].shape} expected={param.shape}"
            )
    return errors
