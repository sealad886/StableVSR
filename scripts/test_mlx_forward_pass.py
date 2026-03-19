"""Forward pass test with real weights for all MLX models."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import mlx.core as mx
import mlx.nn as nn

MODEL_PATH = Path(
    "models/StableVSR/models--claudiom4sir--StableVSR/snapshots/"
    "fddd0e3921c22a5dcc6468c56c44abe6564bacc2"
)


def load_model_and_weights(
    model_cls, config_path, weights_path, extra_kwargs=None, prefix_strip=None
):
    """Helper to load config, create model, load weights."""
    from stablevsr.mlx.weight_utils import load_safetensors_for_mlx

    with open(config_path) as f:
        cfg = json.load(f)

    kwargs = extra_kwargs or {}
    model = model_cls(**kwargs)

    weights = load_safetensors_for_mlx(weights_path, config_path)
    if prefix_strip:
        weights = {
            (k[len(prefix_strip) :] if k.startswith(prefix_strip) else k): v
            for k, v in weights.items()
        }
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model, cfg


def test_text_encoder_forward():
    from stablevsr.mlx.models.text_encoder import CLIPTextModel

    with open(MODEL_PATH / "text_encoder" / "config.json") as f:
        cfg = json.load(f)

    model = CLIPTextModel(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_hidden_layers=cfg["num_hidden_layers"],
        intermediate_size=cfg["intermediate_size"],
        max_position_embeddings=cfg["max_position_embeddings"],
    )
    from stablevsr.mlx.weight_utils import load_safetensors_for_mlx

    weights = load_safetensors_for_mlx(
        MODEL_PATH / "text_encoder" / "model.safetensors"
    )
    remapped = {
        (k.replace("text_model.", "", 1) if k.startswith("text_model.") else k): v
        for k, v in weights.items()
    }
    model.load_weights(list(remapped.items()), strict=False)
    mx.eval(model.parameters())

    # "a high quality photo" tokenized
    input_ids = mx.array([[49406, 320, 1551, 3226, 1125, 49407] + [49407] * 71])
    out = model(input_ids)
    mx.eval(out)

    assert out.shape == (1, 77, 1024), f"Wrong shape: {out.shape}"
    assert not mx.isnan(out).any().item(), "NaN in text encoder output"
    assert (
        mx.abs(out).max().item() < 1000
    ), f"Text encoder output too large: {mx.abs(out).max().item()}"
    print(
        f"  Text Encoder: shape={out.shape}, range=[{out.min().item():.4f}, {out.max().item():.4f}]"
    )
    return True


def test_vae_forward():
    from stablevsr.mlx.models.vae import AutoencoderKL
    from stablevsr.mlx.weight_utils import load_safetensors_for_mlx

    with open(MODEL_PATH / "vae" / "config.json") as f:
        cfg = json.load(f)

    model = AutoencoderKL(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 3),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg.get("layers_per_block", 2),
        latent_channels=cfg.get("latent_channels", 4),
        norm_num_groups=cfg.get("norm_num_groups", 32),
        scaling_factor=cfg.get("scaling_factor", 0.08333),
    )
    weights = load_safetensors_for_mlx(
        MODEL_PATH / "vae" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "vae" / "config.json",
    )
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Test encode
    img = mx.random.normal((1, 64, 64, 3)) * 0.5
    latent = model.encode(img)
    mx.eval(latent)
    print(
        f"  VAE Encode: {img.shape} -> {latent.shape}, range=[{latent.min().item():.4f}, {latent.max().item():.4f}]"
    )
    assert not mx.isnan(latent).any().item(), "NaN in VAE encode"

    # Test decode
    z = mx.random.normal((1, 16, 16, 4)) * 0.1
    decoded = model.decode(z)
    mx.eval(decoded)
    print(
        f"  VAE Decode: {z.shape} -> {decoded.shape}, range=[{decoded.min().item():.4f}, {decoded.max().item():.4f}]"
    )
    assert not mx.isnan(decoded).any().item(), "NaN in VAE decode"

    # Roundtrip test
    z2 = model.encode(decoded)
    mx.eval(z2)
    print(
        f"  VAE Roundtrip: latent range=[{z2.min().item():.4f}, {z2.max().item():.4f}]"
    )
    return True


def test_unet_forward():
    from stablevsr.mlx.models.unet import UNet2DConditionModel
    from stablevsr.mlx.weight_utils import load_safetensors_for_mlx

    with open(MODEL_PATH / "unet" / "config.json") as f:
        cfg = json.load(f)

    model = UNet2DConditionModel(
        in_channels=cfg["in_channels"],
        out_channels=cfg.get("out_channels", 4),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg.get("layers_per_block", 2),
        cross_attention_dim=cfg.get("cross_attention_dim", 1024),
        attention_head_dim=cfg.get("attention_head_dim", 8),
        only_cross_attention=tuple(cfg.get("only_cross_attention", [False] * 4)),
        down_block_types=tuple(
            cfg.get(
                "down_block_types",
                [
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                ],
            )
        ),
        up_block_types=tuple(
            cfg.get(
                "up_block_types",
                [
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",
                ],
            )
        ),
    )
    weights = load_safetensors_for_mlx(
        MODEL_PATH / "unet" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "unet" / "config.json",
    )
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Small test: 8x8 spatial, batch=1
    sample = mx.random.normal((1, 8, 8, 7))  # 7 channels (4 latent + 3 image)
    timestep = mx.array([500])
    encoder_hidden = mx.random.normal((1, 77, 1024))

    out = model(sample, timestep, encoder_hidden)
    mx.eval(out)

    print(
        f"  UNet: {sample.shape} -> {out.shape}, range=[{out.min().item():.4f}, {out.max().item():.4f}]"
    )
    assert out.shape == (1, 8, 8, 4), f"Wrong UNet output shape: {out.shape}"
    assert not mx.isnan(out).any().item(), "NaN in UNet output"
    return True


def test_controlnet_forward():
    from stablevsr.mlx.models.controlnet import ControlNetModel
    from stablevsr.mlx.weight_utils import load_safetensors_for_mlx

    with open(MODEL_PATH / "controlnet" / "config.json") as f:
        cfg = json.load(f)

    model = ControlNetModel(
        in_channels=cfg["in_channels"],
        conditioning_channels=cfg.get("conditioning_channels", 3),
        block_out_channels=tuple(cfg["block_out_channels"]),
        layers_per_block=cfg.get("layers_per_block", 2),
        cross_attention_dim=cfg.get("cross_attention_dim", 1024),
        attention_head_dim=cfg.get("attention_head_dim", 8),
        only_cross_attention=tuple(
            cfg.get("only_cross_attention", [True, True, True, False])
        ),
        conditioning_embedding_out_channels=tuple(
            cfg.get("conditioning_embedding_out_channels", [64, 128, 256])
        ),
        down_block_types=tuple(
            cfg.get(
                "down_block_types",
                [
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                ],
            )
        ),
    )
    weights = load_safetensors_for_mlx(
        MODEL_PATH / "controlnet" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "controlnet" / "config.json",
    )
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Small test
    sample = mx.random.normal((1, 8, 8, 7))
    timestep = mx.array([500])
    encoder_hidden = mx.random.normal((1, 77, 1024))
    cond = mx.random.normal((1, 32, 32, 3))  # 4x spatial (vae_scale=4)

    down_res, mid_res = model(sample, timestep, encoder_hidden, cond)
    mx.eval(down_res)
    mx.eval(mid_res)

    print(f"  ControlNet: {len(down_res)} down residuals, mid={mid_res.shape}")
    for i, r in enumerate(down_res):
        print(f"    down[{i}]: {r.shape}")
    assert not mx.isnan(mid_res).any().item(), "NaN in ControlNet mid output"
    return True


if __name__ == "__main__":
    tests = [
        ("Text Encoder", test_text_encoder_forward),
        ("VAE", test_vae_forward),
        ("UNet", test_unet_forward),
        ("ControlNet", test_controlnet_forward),
    ]

    results = []
    for name, fn in tests:
        print(f"\n--- {name} ---")
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            print(f"  RESULT: {name}: PASSED ({elapsed:.1f}s)")
            results.append((name, True))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  RESULT: {name}: FAILED ({elapsed:.1f}s): {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{'='*40}")
    print(f"{passed}/{len(results)} forward pass tests passed")
    sys.exit(0 if passed == len(results) else 1)
