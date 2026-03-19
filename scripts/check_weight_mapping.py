"""Verify weight mapping completeness between safetensors and MLX models."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

MODEL_PATH = Path(
    "models/StableVSR/models--claudiom4sir--StableVSR/snapshots/"
    "fddd0e3921c22a5dcc6468c56c44abe6564bacc2"
)


def get_model_param_keys(model):
    """Get all parameter keys from an MLX model."""
    return {k for k, _ in nn.utils.tree_flatten(model.parameters())}


def get_safetensors_keys(path):
    """Get all keys from safetensors file."""
    with safe_open(str(path), framework="numpy") as f:
        return set(f.keys())


def check_mapping(name, model, safetensors_path, config_path=None, prefix_strip=None):
    """Check how safetensors keys map to model parameters."""
    from stablevsr.mlx.weight_utils import load_safetensors_for_mlx

    model_keys = get_model_param_keys(model)
    sf_keys = get_safetensors_keys(safetensors_path)

    # Load weights through our mapping
    weights = load_safetensors_for_mlx(safetensors_path, config_path)

    if prefix_strip:
        weights = {
            (k[len(prefix_strip):] if k.startswith(prefix_strip) else k): v
            for k, v in weights.items()
        }

    weight_keys = set(weights.keys())

    loaded_in_model = model_keys & weight_keys
    missing_in_weights = model_keys - weight_keys
    extra_in_weights = weight_keys - model_keys

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Safetensors keys:    {len(sf_keys)}")
    print(f"  After mapping:       {len(weight_keys)}")
    print(f"  Model params:        {len(model_keys)}")
    print(f"  Loaded into model:   {len(loaded_in_model)}")
    print(f"  Missing in weights:  {len(missing_in_weights)}")
    print(f"  Extra in weights:    {len(extra_in_weights)}")

    if missing_in_weights:
        print(f"\n  MISSING (model needs but weights lack):")
        for k in sorted(missing_in_weights)[:20]:
            print(f"    - {k}")
        if len(missing_in_weights) > 20:
            print(f"    ... and {len(missing_in_weights) - 20} more")

    if extra_in_weights:
        print(f"\n  EXTRA (weights have but model doesn't):")
        for k in sorted(extra_in_weights)[:20]:
            print(f"    - {k}")
        if len(extra_in_weights) > 20:
            print(f"    ... and {len(extra_in_weights) - 20} more")

    return len(missing_in_weights), len(extra_in_weights)


def main():
    from stablevsr.mlx.models.text_encoder import CLIPTextModel
    from stablevsr.mlx.models.vae import AutoencoderKL
    from stablevsr.mlx.models.unet import UNet2DConditionModel
    from stablevsr.mlx.models.controlnet import ControlNetModel

    total_missing = 0
    total_extra = 0

    # Text Encoder
    with open(MODEL_PATH / "text_encoder" / "config.json") as f:
        cfg = json.load(f)
    te = CLIPTextModel(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_hidden_layers=cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        max_position_embeddings=cfg["max_position_embeddings"],
    )
    m, e = check_mapping(
        "Text Encoder", te,
        MODEL_PATH / "text_encoder" / "model.safetensors",
        prefix_strip="text_model.",
    )
    total_missing += m
    total_extra += e

    # VAE
    with open(MODEL_PATH / "vae" / "config.json") as f:
        cfg = json.load(f)
    vae = AutoencoderKL(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 3),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg.get("layers_per_block", 2),
        latent_channels=cfg.get("latent_channels", 4),
        norm_num_groups=cfg.get("norm_num_groups", 32),
        scaling_factor=cfg.get("scaling_factor", 0.08333),
    )
    m, e = check_mapping(
        "VAE", vae,
        MODEL_PATH / "vae" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "vae" / "config.json",
    )
    total_missing += m
    total_extra += e

    # UNet
    with open(MODEL_PATH / "unet" / "config.json") as f:
        cfg = json.load(f)
    unet = UNet2DConditionModel(
        in_channels=cfg["in_channels"],
        out_channels=cfg.get("out_channels", 4),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg.get("layers_per_block", 2),
        cross_attention_dim=cfg.get("cross_attention_dim", 1024),
        attention_head_dim=cfg.get("attention_head_dim", 8),
        only_cross_attention=tuple(cfg.get("only_cross_attention", [False] * 4)),
    )
    m, e = check_mapping(
        "UNet", unet,
        MODEL_PATH / "unet" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "unet" / "config.json",
    )
    total_missing += m
    total_extra += e

    # ControlNet
    with open(MODEL_PATH / "controlnet" / "config.json") as f:
        cfg = json.load(f)
    cn = ControlNetModel(
        in_channels=cfg["in_channels"],
        conditioning_channels=cfg.get("conditioning_channels", 3),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg.get("layers_per_block", 2),
        cross_attention_dim=cfg.get("cross_attention_dim", 1024),
        attention_head_dim=cfg.get("attention_head_dim", 8),
        only_cross_attention=tuple(cfg.get("only_cross_attention", [True, True, True, False])),
    )
    m, e = check_mapping(
        "ControlNet", cn,
        MODEL_PATH / "controlnet" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "controlnet" / "config.json",
    )
    total_missing += m
    total_extra += e

    print(f"\n{'='*60}")
    print(f"  TOTAL: {total_missing} missing, {total_extra} extra")
    print(f"{'='*60}")
    return total_missing


if __name__ == "__main__":
    missing = main()
    sys.exit(1 if missing > 0 else 0)
