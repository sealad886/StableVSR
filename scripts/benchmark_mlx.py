"""Benchmark matrix for MLX StableVSR pipeline on Apple Silicon.

Runs configurable test scenarios and outputs structured JSON + markdown results.
Captures: model load time, per-stage timing, peak memory, and optimization state.
"""

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("benchmark")

MODEL_PATH = Path(
    "models/StableVSR/models--claudiom4sir--StableVSR/snapshots/"
    "fddd0e3921c22a5dcc6468c56c44abe6564bacc2"
)
FRAME_DIR = Path("e2e_test/input/frontdoor_clip")

SCENARIOS = [
    {
        "name": "smoke-160x90-1f-notile",
        "resolution": (160, 90),
        "frames": 1, "steps": 2,
        "force_tiled": False, "dtype": "float16",
        "compile": False, "ttg_start": 0,
    },
    {
        "name": "smoke-160x90-1f-tiled",
        "resolution": (160, 90),
        "frames": 1, "steps": 2,
        "force_tiled": True, "dtype": "float16",
        "compile": False, "ttg_start": 0,
    },
    {
        "name": "native-480x270-2f-tiled",
        "resolution": None,
        "frames": 2, "steps": 2,
        "force_tiled": None, "dtype": "float16",
        "compile": False, "ttg_start": 0,
    },
    {
        "name": "native-480x270-2f-compiled",
        "resolution": None,
        "frames": 2, "steps": 2,
        "force_tiled": None, "dtype": "float16",
        "compile": True, "ttg_start": 0,
    },
    {
        "name": "native-480x270-2f-ttg1",
        "resolution": None,
        "frames": 2, "steps": 2,
        "force_tiled": None, "dtype": "float16",
        "compile": False, "ttg_start": 1,
    },
    {
        "name": "native-480x270-2f-compiled-ttg1",
        "resolution": None,
        "frames": 2, "steps": 2,
        "force_tiled": None, "dtype": "float16",
        "compile": True, "ttg_start": 1,
    },
    {
        "name": "native-480x270-2f-notile",
        "resolution": None,
        "frames": 2, "steps": 2,
        "force_tiled": False, "dtype": "float16",
        "compile": False, "ttg_start": 0,
    },
    {
        "name": "native-480x270-2f-5step",
        "resolution": None,
        "frames": 2, "steps": 5,
        "force_tiled": None, "dtype": "float16",
        "compile": False, "ttg_start": 0,
    },
    {
        "name": "native-480x270-5f-tiled",
        "resolution": None,
        "frames": 5, "steps": 2,
        "force_tiled": None, "dtype": "float16",
        "compile": False, "ttg_start": 0,
    },
    {
        "name": "native-480x270-2f-fp32",
        "resolution": None,
        "frames": 2, "steps": 2,
        "force_tiled": None, "dtype": "float32",
        "compile": False, "ttg_start": 0,
    },
]


def load_frames(n_frames: int, resolution: tuple[int, int] | None) -> list[np.ndarray]:
    """Load and optionally resize test frames."""
    frame_paths = sorted(FRAME_DIR.glob("*.png"))
    if len(frame_paths) < n_frames:
        # Repeat frames if needed
        frame_paths = frame_paths * ((n_frames // len(frame_paths)) + 1)
    frame_paths = frame_paths[:n_frames]

    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        if resolution is not None:
            img = img.resize(resolution, Image.LANCZOS)
        frames.append(np.array(img))
    return frames


def load_raft():
    """Load RAFT optical flow model."""
    from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

    model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    model = model.eval().cpu()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_scenario(scenario: dict, raft_model, output_dir: Path) -> dict:
    """Run a single benchmark scenario and return results."""
    import mlx.core as mx

    name = scenario["name"]
    logger.info("=" * 60)
    logger.info(f"Scenario: {name}")
    logger.info("=" * 60)

    dtype_map = {"float16": mx.float16, "float32": mx.float32}
    dtype = dtype_map[scenario["dtype"]]

    result = {
        "name": name,
        "resolution": scenario["resolution"],
        "frames": scenario["frames"],
        "steps": scenario["steps"],
        "force_tiled": scenario["force_tiled"],
        "dtype": scenario["dtype"],
        "compile": scenario.get("compile", False),
        "ttg_start": scenario.get("ttg_start", 0),
    }

    # Reset memory tracking
    mx.reset_peak_memory()

    # Load pipeline
    t0 = time.time()
    from stablevsr.mlx.pipeline import MLXStableVSRPipeline

    pipe = MLXStableVSRPipeline.from_pretrained(MODEL_PATH, dtype=dtype)
    t_load = time.time() - t0
    result["load_time_s"] = round(t_load, 2)
    result["mem_after_load_gb"] = round(mx.get_peak_memory() / (1024**3), 2)
    logger.info(f"  Load: {t_load:.1f}s, {result['mem_after_load_gb']:.2f} GB")

    # Load frames
    frames = load_frames(scenario["frames"], scenario["resolution"])
    actual_res = f"{frames[0].shape[1]}x{frames[0].shape[0]}"
    result["actual_resolution"] = actual_res
    logger.info(f"  Frames: {len(frames)} @ {actual_res}")

    # Build call kwargs
    call_kwargs = {
        "images": frames,
        "of_model": raft_model,
        "num_inference_steps": scenario["steps"],
        "guidance_scale": 7.5,
        "controlnet_conditioning_scale": 1.0,
        "seed": 42,
        "compile_models": scenario.get("compile", False),
        "ttg_start_step": scenario.get("ttg_start", 0),
    }
    if scenario["force_tiled"] is not None:
        call_kwargs["force_tiled_vae"] = scenario["force_tiled"]

    # Run pipeline
    mx.reset_peak_memory()
    t_start = time.time()
    try:
        output_frames = pipe(**call_kwargs)
        t_total = time.time() - t_start
        result["status"] = "success"
        result["total_time_s"] = round(t_total, 2)
        result["peak_mem_gb"] = round(mx.get_peak_memory() / (1024**3), 2)
        result["output_shape"] = list(output_frames[0].shape)
        logger.info(
            f"  Done: {t_total:.1f}s, peak={result['peak_mem_gb']:.2f} GB, "
            f"output={output_frames[0].shape}"
        )

        # Save first output frame
        out_path = output_dir / f"{name}.png"
        Image.fromarray(output_frames[0]).save(out_path)

    except Exception as e:
        t_total = time.time() - t_start
        result["status"] = f"error: {type(e).__name__}: {e}"
        result["total_time_s"] = round(t_total, 2)
        result["peak_mem_gb"] = round(mx.get_peak_memory() / (1024**3), 2)
        logger.error(f"  FAILED: {e}")

    # Cleanup
    del pipe
    gc.collect()

    return result


def generate_markdown(results: list[dict]) -> str:
    """Generate markdown report from results."""
    lines = [
        "# MLX StableVSR Benchmark Results",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Platform**: Apple Silicon (MLX {_get_mlx_version()})",
        "",
        "## Results",
        "",
        "| Scenario | Resolution | Frames | Steps | Dtype | Tiled | Time (s) | Peak Mem (GB) | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tiled = {True: "forced", False: "off", None: "auto"}.get(
            r.get("force_tiled"), "auto"
        )
        lines.append(
            f"| {r['name']} | {r.get('actual_resolution', 'N/A')} | "
            f"{r['frames']} | {r['steps']} | {r['dtype']} | {tiled} | "
            f"{r.get('total_time_s', 'N/A')} | {r.get('peak_mem_gb', 'N/A')} | "
            f"{r['status']} |"
        )

    lines.extend([
        "",
        "## Configuration",
        "",
        f"- Model: `{MODEL_PATH}`",
        f"- Frames source: `{FRAME_DIR}`",
        "- Guidance scale: 7.5",
        "- Seed: 42",
        "",
        "## Notes",
        "",
        "- **auto** tiling: activates when latent area > 4096 pixels (64×64)",
        "- **forced** tiling: tile_size=64, tile_overlap=16 (defaults)",
        "- OOM scenarios will show as `error` status with memory at time of failure",
        "- Load time includes model weight loading and dtype conversion",
    ])
    return "\n".join(lines)


def _get_mlx_version() -> str:
    try:
        import mlx

        return mlx.__version__
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="MLX StableVSR benchmark matrix")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Run only named scenarios (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("e2e_test/output/benchmark"),
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Path to write JSON results (default: <output-dir>/results.json)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_out = args.json_out or args.output_dir / "results.json"

    # Filter scenarios
    if args.scenarios is not None and len(args.scenarios) > 0:
        scenarios = [s for s in SCENARIOS if s["name"] in args.scenarios]
    else:
        scenarios = SCENARIOS

    logger.info(f"Running {len(scenarios)} benchmark scenarios")

    # Load RAFT once
    logger.info("Loading RAFT model...")
    raft_model = load_raft()

    results = []
    for scenario in scenarios:
        result = run_scenario(scenario, raft_model, args.output_dir)
        results.append(result)

    # Write JSON
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"JSON results written to {json_out}")

    # Write markdown
    md = generate_markdown(results)
    md_path = args.output_dir / "benchmark_results.md"
    with open(md_path, "w") as f:
        f.write(md)
    logger.info(f"Markdown report written to {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for r in results:
        status = "OK" if r["status"] == "success" else "FAIL"
        t = r.get("total_time_s", "N/A")
        m = r.get("peak_mem_gb", "N/A")
        print(f"  {r['name']:40s} {status:4s}  {t:>7}s  {m:>6} GB")


if __name__ == "__main__":
    main()
