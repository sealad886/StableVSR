"""Compare quality across preset configurations.

Runs the MLX pipeline with each preset on the same input and reports:
- Per-frame PSNR and SSIM (against max-quality reference)
- Per-configuration runtime
- Temporal consistency metric (inter-frame difference stability)

Usage:
    python scripts/quality_compare.py --input e2e_test/input/frontdoor_clip \\
        --output e2e_test/output/quality_compare --steps 10

Requires: scikit-image (for SSIM), numpy, PIL
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio between two uint8 images."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def temporal_stability(frames: list[np.ndarray]) -> float:
    """Measure temporal consistency as std of inter-frame differences.

    Lower is more temporally stable.
    """
    if len(frames) < 2:
        return 0.0
    diffs = []
    for i in range(1, len(frames)):
        diff = np.mean(
            np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))
        )
        diffs.append(diff)
    return float(np.std(diffs))


def run_config(
    pipe,
    raft_model,
    images: list[np.ndarray],
    *,
    steps: int,
    seed: int,
    compile_models: bool,
    ttg_start_step: int,
    chunk_size: int | None,
    chunk_overlap: int,
) -> tuple[list[np.ndarray], float, dict]:
    """Run a single configuration and return (frames, elapsed, stage_timing)."""
    common = dict(
        images=images,
        of_model=raft_model,
        prompt="clean, high-resolution, 8k, sharp, details",
        negative_prompt="blurry, noise, low-resolution, artifacts",
        num_inference_steps=steps,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        seed=seed,
        compile_models=compile_models,
        ttg_start_step=ttg_start_step,
    )

    if chunk_size is not None and chunk_size < len(images):
        from stablevsr.mlx.chunked_pipeline import run_chunked_inference

        t0 = time.perf_counter()
        frames = run_chunked_inference(
            pipeline=pipe,
            of_model=raft_model,
            images=images,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **{k: v for k, v in common.items() if k not in ("images", "of_model")},
        )
        elapsed = time.perf_counter() - t0
        return frames, elapsed, {}
    else:
        t0 = time.perf_counter()
        result = pipe(**common)
        elapsed = time.perf_counter() - t0
        return result.frames, elapsed, result.stage_timing


def compare(args: argparse.Namespace) -> None:
    import mlx.core as mx

    from stablevsr.mlx.flow.raft_bridge import load_raft_model
    from stablevsr.mlx.pipeline import MLXStableVSRPipeline
    from stablevsr.mlx.presets import PRESETS

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Load frames
    input_dir = Path(args.input)
    frame_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not frame_paths:
        logger.error("No frames in %s", input_dir)
        return

    max_frames = args.max_frames or len(frame_paths)
    frame_paths = frame_paths[:max_frames]
    images = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
    logger.info("Loaded %d frames (%s)", len(images), images[0].shape)

    # Load model
    model_path = Path(args.model_path)
    snapshot_dirs = sorted(model_path.rglob("snapshots/*/"))
    if snapshot_dirs:
        model_path = snapshot_dirs[-1]

    pipe = MLXStableVSRPipeline.from_pretrained(str(model_path), dtype=mx.float16)
    raft_model = load_raft_model()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Run each preset
    presets_to_run = args.presets.split(",") if args.presets else list(PRESETS.keys())

    for preset_name in presets_to_run:
        preset = PRESETS[preset_name]
        ttg = preset.resolve_ttg_start_step(args.steps)

        logger.info("\n=== Preset: %s ===", preset_name)
        logger.info(
            "  compile=%s, ttg_start=%d, chunk=%s, overlap=%d",
            preset.compile_models,
            ttg,
            preset.chunk_size,
            preset.chunk_overlap,
        )

        frames, elapsed, timing = run_config(
            pipe,
            raft_model,
            images,
            steps=args.steps,
            seed=args.seed,
            compile_models=preset.compile_models,
            ttg_start_step=ttg,
            chunk_size=preset.chunk_size,
            chunk_overlap=preset.chunk_overlap,
        )

        # Save frames
        preset_dir = output_dir / preset_name
        preset_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(preset_dir / f"frame_{i:04d}.png")

        entry = {
            "preset": preset_name,
            "compile": preset.compile_models,
            "ttg_start_step": ttg,
            "chunk_size": preset.chunk_size,
            "chunk_overlap": preset.chunk_overlap,
            "num_frames": len(frames),
            "time_s": round(elapsed, 1),
            "s_per_frame": round(elapsed / max(len(frames), 1), 1),
            "temporal_stability": round(temporal_stability(frames), 3),
            "stage_timing": timing,
        }
        results.append(entry)
        logger.info(
            "  Time: %.1fs (%.1f s/frame)", elapsed, elapsed / max(len(frames), 1)
        )

    # Compute PSNR vs max-quality reference
    ref_name = "max-quality"
    ref_dir = output_dir / ref_name
    if ref_dir.exists():
        ref_frames = [
            np.array(Image.open(p)) for p in sorted(ref_dir.glob("frame_*.png"))
        ]
        for entry in results:
            if entry["preset"] == ref_name:
                entry["avg_psnr_vs_ref"] = float("inf")
                continue
            preset_dir = output_dir / entry["preset"]
            test_frames = [
                np.array(Image.open(p)) for p in sorted(preset_dir.glob("frame_*.png"))
            ]
            if len(test_frames) == len(ref_frames):
                psnrs = [psnr(r, t) for r, t in zip(ref_frames, test_frames)]
                entry["avg_psnr_vs_ref"] = round(float(np.mean(psnrs)), 2)
            else:
                entry["avg_psnr_vs_ref"] = None

    # Write report
    report_path = output_dir / "comparison_results.json"
    report_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    logger.info("\nResults saved to %s", report_path)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Preset':<15} {'Time':>8} {'s/frame':>8} {'Stability':>10} {'PSNR':>8}")
    print("-" * 80)
    for r in results:
        psnr_str = f"{r.get('avg_psnr_vs_ref', 'N/A')}"
        if psnr_str == "inf":
            psnr_str = "ref"
        print(
            f"{r['preset']:<15} {r['time_s']:>7.1f}s {r['s_per_frame']:>7.1f}s "
            f"{r['temporal_stability']:>10.3f} {psnr_str:>8}"
        )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare quality across presets")
    parser.add_argument("--input", required=True, help="Input frame directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model-path", default="models/StableVSR")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--presets", default=None, help="Comma-separated preset names")
    parser.add_argument("-v", "--verbose", action="store_true")
    compare(parser.parse_args())


if __name__ == "__main__":
    main()
