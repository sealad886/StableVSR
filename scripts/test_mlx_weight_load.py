"""Test loading real StableVSR weights into MLX models."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "src")
import mlx.core as mx
import mlx.nn as nn

MODEL_PATH = Path("models/StableVSR/models--claudiom4sir--StableVSR/snapshots/fddd0e3921c22a5dcc6468c56c44abe6564bacc2")


def test_text_encoder():
    from stablevsr.mlx.models.text_encoder import CLIPTextModel
    from stablevsr.mlx.weight_utils import load_safetensors_for_mlx

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

    weights = load_safetensors_for_mlx(MODEL_PATH / "text_encoder" / "model.safetensors")
    remapped = {}
    for k, v in weights.items():
        new_k = k.replace("text_model.", "", 1) if k.startswith("text_model.") else k
        remapped[new_k] = v

    model.load_weights(list(remapped.items()), strict=False)
    mx.eval(model.parameters())

    input_ids = mx.array([[49406, 4480, 267, 3675, 267, 49407] + [49407] * 71])
    out = model(input_ids)
    mx.eval(out)

    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"  Text Encoder: {n_params:,} params, output = {out.shape}, dtype = {out.dtype}")
    assert out.shape == (1, 77, cfg["hidden_size"])
    return True


def test_vae():
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

    z = mx.random.normal((1, 8, 8, 4))
    decoded = model.decode(z)
    mx.eval(decoded)

    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"  VAE: {n_params:,} params, decode = {decoded.shape}, dtype = {decoded.dtype}")
    # 3 blocks → 2 upsamples → 4× spatial factor: 8→32
    assert decoded.shape == (1, 32, 32, 3)
    return True


def test_unet():
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
    )

    weights = load_safetensors_for_mlx(
        MODEL_PATH / "unet" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "unet" / "config.json",
    )
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"  UNet: {n_params:,} params loaded")
    return True


def test_controlnet():
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
        only_cross_attention=tuple(cfg.get("only_cross_attention", [True, True, True, False])),
    )

    weights = load_safetensors_for_mlx(
        MODEL_PATH / "controlnet" / "diffusion_pytorch_model.safetensors",
        MODEL_PATH / "controlnet" / "config.json",
    )
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    print(f"  ControlNet: {n_params:,} params loaded")
    return True


if __name__ == "__main__":
    tests = [
        ("Text Encoder", test_text_encoder),
        ("VAE", test_vae),
        ("UNet", test_unet),
        ("ControlNet", test_controlnet),
    ]

    results = []
    for name, fn in tests:
        t0 = time.time()
        try:
            ok = fn()
            elapsed = time.time() - t0
            status = "PASSED"
            print(f"  RESULT: {name}: {status} ({elapsed:.1f}s)")
            results.append((name, True))
        except Exception as e:
            elapsed = time.time() - t0
            status = "FAILED"
            print(f"  RESULT: {name}: {status} ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} weight loading tests passed")
    sys.exit(0 if passed == len(results) else 1)
