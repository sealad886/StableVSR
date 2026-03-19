"""Profile StableVSR inference using method-level instrumentation.

Wraps key pipeline methods with timing hooks so the actual pipeline.__call__()
runs normally (with CPU offloading, attention slicing, etc.) while capturing
per-stage wall-clock times.

Usage:
    python scripts/profile_inference.py --input e2e_test/input/frontdoor_clip --frames 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _sync_device(device_type: str) -> None:
    import torch

    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


class TimingAccumulator:
    """Accumulate wall-clock seconds for monkey-patched methods."""

    def __init__(self, device_type: str = "cpu"):
        self.device_type = device_type
        self.records: dict[str, list[float]] = defaultdict(list)

    def wrap(self, obj: object, method_name: str, label: str | None = None):
        label = label or method_name
        original = getattr(obj, method_name)

        def timed(*args, **kwargs):
            _sync_device(self.device_type)
            t0 = time.perf_counter()
            result = original(*args, **kwargs)
            _sync_device(self.device_type)
            self.records[label].append(time.perf_counter() - t0)
            return result

        setattr(obj, method_name, timed)

    def summary(self) -> dict[str, dict]:
        out = {}
        for label, times in self.records.items():
            out[label] = {
                "calls": len(times),
                "total_s": round(sum(times), 4),
                "mean_ms": round(1000 * sum(times) / len(times), 2) if times else 0,
                "min_ms": round(1000 * min(times), 2) if times else 0,
                "max_ms": round(1000 * max(times), 2) if times else 0,
            }
        return out


def _get_memory_mb() -> float | None:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Profile StableVSR inference pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--frames", type=int, default=0, help="0=all")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default="claudiom4sir/StableVSR")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    timings: dict[str, float] = {}
    mem: dict[str, float | None] = {}
    t_wall_start = time.perf_counter()
    mem["start"] = _get_memory_mb()

    # --- Imports ---
    t0 = time.perf_counter()
    import torch
    from PIL import Image

    timings["import_core"] = time.perf_counter() - t0

    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    # --- Backend ---
    t0 = time.perf_counter()
    from stablevsr.backends import get_backend

    backend = get_backend()
    device = torch.device(backend.default_device())
    timings["backend_detect"] = time.perf_counter() - t0
    dev_type = device.type
    print(f"Device: {device}, dtype: {args.dtype}, backend: {backend.name()}")

    acc = TimingAccumulator(dev_type)

    # --- Frame load ---
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    input_dir = Path(args.input)
    t0 = time.perf_counter()
    paths = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if args.frames > 0:
        paths = paths[: args.frames]
    frames = [Image.open(p) for p in paths]
    timings["frame_load"] = time.perf_counter() - t0
    n_frames = len(frames)
    w, h = frames[0].size
    print(f"Loaded {n_frames} frames ({w}x{h})")
    mem["after_frame_load"] = _get_memory_mb()

    # --- ControlNet ---
    _sync_device(dev_type)
    t0 = time.perf_counter()
    from diffusers import ControlNetModel, DDPMScheduler

    cn = ControlNetModel.from_pretrained(
        args.model_id, subfolder="controlnet", torch_dtype=dtype
    )
    _sync_device(dev_type)
    timings["load_controlnet"] = time.perf_counter() - t0
    mem["after_controlnet"] = _get_memory_mb()

    # --- Pipeline ---
    _sync_device(dev_type)
    t0 = time.perf_counter()
    from pipeline.stablevsr_pipeline import StableVSRPipeline

    pipe = StableVSRPipeline.from_pretrained(
        args.model_id, controlnet=cn, torch_dtype=dtype
    )
    pipe.scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    _sync_device(dev_type)
    timings["load_pipeline"] = time.perf_counter() - t0
    mem["after_pipeline_load"] = _get_memory_mb()

    # --- Config for MPS ---
    t0 = time.perf_counter()
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    timings["pipeline_config"] = time.perf_counter() - t0
    mem["after_config"] = _get_memory_mb()

    # --- RAFT ---
    _sync_device(dev_type)
    t0 = time.perf_counter()
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    of_model = of_model.to(device).float()
    _sync_device(dev_type)
    timings["load_raft"] = time.perf_counter() - t0
    mem["after_raft"] = _get_memory_mb()

    # --- Instrument methods ---
    acc.wrap(pipe.unet, "forward", "unet_forward")
    acc.wrap(pipe.controlnet, "forward", "controlnet_forward")
    acc.wrap(pipe.vae, "decode", "vae_decode")
    acc.wrap(pipe.vae, "encode", "vae_encode")
    acc.wrap(pipe.text_encoder, "forward", "text_encoder")
    acc.wrap(pipe.scheduler, "step", "scheduler_step")
    import util.flow_utils as of_mod

    acc.wrap(of_mod, "get_flow", "raft_get_flow")
    acc.wrap(of_mod, "flow_warp", "flow_warp")

    # --- Pipeline inference ---
    torch.manual_seed(args.seed)
    _sync_device(dev_type)
    t0 = time.perf_counter()
    result = pipe(
        prompt="",
        images=frames,
        num_inference_steps=args.steps,
        guidance_scale=1.0,
        output_type="pil",
        of_model=of_model,
        of_rescale_factor=1,
    )
    _sync_device(dev_type)
    timings["pipeline_call"] = time.perf_counter() - t0
    mem["after_inference"] = _get_memory_mb()

    # --- Save ---
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        for i, imgs in enumerate(result.images):
            if isinstance(imgs, list):
                imgs[0].save(out_dir / f"frame_{i:04d}.png")
            else:
                imgs.save(out_dir / f"frame_{i:04d}.png")
        timings["save"] = time.perf_counter() - t0

    t_wall = time.perf_counter() - t_wall_start
    timings["total_wall"] = t_wall
    mem["end"] = _get_memory_mb()

    # --- Report ---
    method_stats = acc.summary()
    report = {
        "config": {
            "frames": n_frames,
            "steps": args.steps,
            "dtype": args.dtype,
            "device": str(device),
            "backend": backend.name(),
            "input_res": f"{w}x{h}",
            "output_res": f"{w * 4}x{h * 4}",
        },
        "timings_s": timings,
        "method_stats": method_stats,
        "memory_mb": mem,
    }

    print("\n" + "=" * 70)
    print("PROFILING REPORT")
    print("=" * 70)
    print(f"Frames: {n_frames} @ {w}x{h} -> {w * 4}x{h * 4}")
    print(f"Steps: {args.steps}, dtype: {args.dtype}, device: {device}")
    print()

    print("TOP-LEVEL TIMINGS:")
    for k, v in timings.items():
        pct = 100 * v / t_wall if t_wall > 0 else 0
        print(f"  {k:30s}  {v:8.3f}s  ({pct:5.1f}%)")

    print()
    print("METHOD-LEVEL BREAKDOWN:")
    sorted_m = sorted(method_stats.items(), key=lambda x: x[1]["total_s"], reverse=True)
    for label, s in sorted_m:
        print(
            f"  {label:25s}  "
            f"{s['mean_ms']:8.1f} ms/call  "
            f"x{s['calls']:4d}  "
            f"total={s['total_s']:8.3f}s"
        )

    print()
    print("MEMORY (peak RSS MB):")
    for k, v in mem.items():
        if v is not None:
            print(f"  {k:30s}  {v:,.0f} MB")
        else:
            print(f"  {k:30s}  N/A")

    if args.output:
        json_path = Path(args.output) / "profile_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON: {json_path}")

    print("\n--- JSON ---", file=sys.stderr)
    json.dump(report, sys.stderr, indent=2)


if __name__ == "__main__":
    main()
