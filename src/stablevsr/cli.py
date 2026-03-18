"""StableVSR command-line interface."""

from __future__ import annotations

import argparse
import logging
import platform
import struct
import sys
from pathlib import Path

from stablevsr import __version__

log = logging.getLogger(__name__)


def cmd_backend_info(args: argparse.Namespace) -> None:
    """Print backend detection results."""
    from stablevsr.backends import list_backends

    for caps in list_backends():
        status = "AVAILABLE" if caps.available else "NOT AVAILABLE"
        print(f"\n[{caps.name}] {status}")
        print(f"  Device:         {caps.device_name or 'N/A'}")
        print(f"  Inference:      {'yes' if caps.inference else 'no'}")
        print(f"  Training:       {'yes' if caps.training else 'no'}")
        print(f"  Half precision: {'yes' if caps.half_precision else 'no'}")
        for note in caps.notes:
            print(f"  Note: {note}")


def _detect_platform_info() -> dict[str, str]:
    """Gather platform information for diagnostics."""
    info: dict[str, str] = {}
    info["os"] = platform.system()
    info["os_version"] = (
        platform.mac_ver()[0] if platform.system() == "Darwin" else platform.version()
    )
    info["arch"] = platform.machine()
    info["python"] = platform.python_version()
    info["pointer_size"] = str(struct.calcsize("P") * 8)

    if platform.system() == "Darwin":
        info["is_arm64"] = "yes" if platform.machine() == "arm64" else "no"
        info["is_rosetta"] = "no"
        if platform.machine() == "x86_64":
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "sysctl.proc_translated"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip() == "1":
                    info["is_rosetta"] = "yes"
            except Exception:
                info["is_rosetta"] = "unknown"
    return info


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run diagnostic checks."""
    issues: list[str] = []

    # Platform info
    plat = _detect_platform_info()
    print(f"[INFO] Platform: {plat['os']} {plat['os_version']} ({plat['arch']})")
    print(f"[INFO] Python: {plat['python']} ({plat['pointer_size']}-bit)")
    if plat.get("is_arm64") == "yes":
        print("[OK] Native Apple Silicon (arm64)")
    elif plat.get("is_rosetta") == "yes":
        print("[WARN] Running under Rosetta 2 (x86_64 translation)")
        issues.append("Running under Rosetta — native arm64 Python recommended")
    elif plat["os"] == "Darwin" and plat["arch"] == "x86_64":
        print("[INFO] Intel Mac (x86_64)")

    # Check PyTorch
    try:
        import torch

        print(f"[OK] PyTorch {torch.__version__}")
        if torch.backends.mps.is_available():
            print("[OK] MPS backend available")
        elif torch.cuda.is_available():
            print(f"[OK] CUDA available ({torch.cuda.get_device_name(0)})")
        else:
            print("[WARN] No GPU backend — CPU only")
            issues.append("No GPU backend detected")
    except ImportError:
        print("[FAIL] PyTorch not installed")
        issues.append("PyTorch not installed")

    # Check diffusers
    try:
        import diffusers

        print(f"[OK] diffusers {diffusers.__version__}")
    except ImportError:
        print("[FAIL] diffusers not installed")
        issues.append("diffusers not installed")

    # Check transformers
    try:
        import transformers

        print(f"[OK] transformers {transformers.__version__}")
    except ImportError:
        print("[FAIL] transformers not installed")
        issues.append("transformers not installed")

    # Check RAFT (torchvision)
    try:
        from torchvision.models.optical_flow import raft_large  # noqa: F401

        print("[OK] RAFT optical flow model available")
    except ImportError:
        print("[FAIL] torchvision RAFT not available")
        issues.append("torchvision RAFT not available")

    # Check video I/O
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[OK] imageio + ffmpeg available ({ffmpeg_path})")
    except ImportError:
        print("[WARN] imageio/ffmpeg not installed (needed for video I/O)")
    except Exception:
        print("[WARN] imageio installed but ffmpeg binary not found")

    # Check MLX (optional)
    try:
        import mlx.core  # noqa: F401

        print("[OK] MLX available (scaffold only — use torch-mps for inference)")
    except ImportError:
        print("[INFO] MLX not installed (optional)")

    if issues:
        print(f"\n{len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("\nAll checks passed.")


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_sequences(input_dir: Path) -> list[tuple[str, Path]]:
    """Return (name, directory) pairs for each image sequence under *input_dir*.

    If *input_dir* contains subdirectories, each is treated as one sequence.
    If it contains images directly (no subdirectories), the whole folder is a
    single sequence.
    """
    subdirs = sorted(
        p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if subdirs:
        return [(d.name, d) for d in subdirs]
    # Flat folder — treat as a single unnamed sequence
    return [(input_dir.name, input_dir)]


def _load_frames(seq_dir: Path) -> tuple[list, list[str]]:
    """Load sorted image files from *seq_dir* as PIL Images."""
    from PIL import Image

    paths = sorted(
        p
        for p in seq_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return [Image.open(p) for p in paths], [p.name for p in paths]


VALID_DTYPES = {"float32", "float16", "bfloat16"}
DTYPE_RESTRICTIONS = {
    "cpu": {"float32"},
    "cuda": {"float32", "float16", "bfloat16"},
    "mps": {"float32", "float16"},
}


def _resolve_dtype(dtype_str: str | None, fp16_flag: bool, device_type: str) -> str:
    """Resolve the effective dtype string, honoring device restrictions."""
    if dtype_str and fp16_flag:
        print("[WARN] Both --dtype and --fp16 specified; --dtype takes precedence")

    if dtype_str is None:
        dtype_str = "float16" if fp16_flag else "float32"

    allowed = DTYPE_RESTRICTIONS.get(device_type, VALID_DTYPES)
    if dtype_str not in allowed:
        print(
            f"[WARN] dtype '{dtype_str}' not supported on {device_type}; "
            f"falling back to float32 (allowed: {', '.join(sorted(allowed))})"
        )
        dtype_str = "float32"

    log.info("Effective dtype: %s (device: %s)", dtype_str, device_type)
    return dtype_str


def _dtype_str_to_torch(dtype_str: str):
    """Convert a dtype string to a torch dtype."""
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_str]


def _run_smoke_test(args: argparse.Namespace) -> None:
    """Validate CLI args and backend selection without loading models."""
    import torch

    from stablevsr.backends import get_backend

    backend = get_backend(args.backend)
    device = torch.device(backend.default_device())
    dtype_str = _resolve_dtype(args.dtype, args.fp16, device.type)

    print("Smoke test passed:")
    print(f"  Backend:   {backend.name()}")
    print(f"  Device:    {device}")
    print(f"  Dtype:     {dtype_str}")
    print(f"  Model ID:  {args.model_id}")
    print(f"  Steps:     {args.steps}")
    print(f"  Seed:      {args.seed}")

    input_dir = Path(args.input)
    if input_dir.is_dir():
        sequences = _collect_sequences(input_dir)
        print(f"  Sequences: {len(sequences)}")
    else:
        print(f"  Input:     {args.input} (not found — would fail at runtime)")


def cmd_infer(args: argparse.Namespace) -> None:
    """Run video super-resolution inference."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    if args.smoke_test:
        _run_smoke_test(args)
        return

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is required for inference. Install it first.")
        sys.exit(1)

    try:
        from diffusers import ControlNetModel, DDPMScheduler
    except ImportError:
        print("[ERROR] diffusers is required for inference. Install it first.")
        sys.exit(1)

    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    except ImportError:
        print("[ERROR] torchvision with RAFT support is required. Install it first.")
        sys.exit(1)

    try:
        from pipeline.stablevsr_pipeline import StableVSRPipeline
    except ImportError:
        print("[ERROR] StableVSRPipeline not found. Run from the project root.")
        sys.exit(1)

    from stablevsr.backends import get_backend

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"[ERROR] Input path does not exist or is not a directory: {input_dir}")
        sys.exit(1)

    # Backend / device
    backend = get_backend(args.backend)
    device = torch.device(backend.default_device())
    device_type = device.type
    dtype_str = _resolve_dtype(args.dtype, args.fp16, device_type)
    dtype = _dtype_str_to_torch(dtype_str)
    print(f"Using backend: {backend.name()} (device: {device}, dtype: {dtype_str})")

    # Seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    model_id = args.model_id
    controlnet_src = args.controlnet_ckpt if args.controlnet_ckpt else model_id
    print(f"Loading controlnet from: {controlnet_src}")
    controlnet_model = ControlNetModel.from_pretrained(
        controlnet_src, subfolder="controlnet", torch_dtype=dtype
    )

    print(f"Loading pipeline from: {model_id}")
    pipeline = StableVSRPipeline.from_pretrained(
        model_id, controlnet=controlnet_model, torch_dtype=dtype
    )
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline.scheduler = scheduler
    pipeline = pipeline.to(device)

    # Memory optimizations — auto-enable MPS safety defaults
    if device.type == "mps":
        if not args.vae_tiling:
            pipeline.enable_vae_tiling()
            print("VAE tiling auto-enabled (MPS)")
        if not args.cpu_offload:
            pipeline.enable_model_cpu_offload()
            print("CPU offload auto-enabled (MPS)")
        pipeline.enable_attention_slicing()
        print("Attention slicing auto-enabled (MPS)")

    if args.vae_tiling:
        pipeline.enable_vae_tiling()
        print("VAE tiling enabled")
    if args.vae_slicing:
        pipeline.enable_vae_slicing()
        print("VAE slicing enabled")
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
        print("Model CPU offload enabled")

    # xformers — only when available and on CUDA
    if device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory-efficient attention")
        except Exception:
            pass

    # Optical flow model — RAFT must stay float32 on MPS
    print("Loading RAFT optical-flow model")
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    of_model = of_model.to(device, dtype=torch.float32)

    # Collect sequences
    sequences = _collect_sequences(input_dir)
    total = len(sequences)
    print(f"Found {total} sequence(s) in {input_dir}")

    for idx, (seq_name, seq_dir) in enumerate(sequences, 1):
        print(f"Processing sequence {idx}/{total}: {seq_name}")
        frames, frame_names = _load_frames(seq_dir)
        if not frames:
            print(f"  Skipping {seq_name} (no images found)")
            continue

        sr_frames = pipeline(
            "",
            frames,
            num_inference_steps=args.steps,
            guidance_scale=0,
            of_model=of_model,
        ).images
        sr_frames = [f[0] for f in sr_frames]

        target_dir = output_dir / seq_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for frame, name in zip(sr_frames, frame_names):
            frame.save(target_dir / name)
        print(f"  Saved {len(sr_frames)} frames to {target_dir}")

    print("Done.")


def _sync(device_type: str) -> None:
    """Block until the accelerator finishes outstanding work."""
    import torch

    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def _peak_rss_mb() -> float | None:
    """Return peak RSS in MiB (macOS/Linux) or *None* on failure."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    except Exception:
        return None


def cmd_benchmark(args: argparse.Namespace) -> None:  # noqa: C901
    """Run a synthetic benchmark of the full StableVSR pipeline."""
    import json
    import time

    import numpy as np
    import torch
    from PIL import Image

    # Ensure project root is importable for pipeline package.
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from diffusers import ControlNetModel, DDPMScheduler
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    from pipeline.stablevsr_pipeline import StableVSRPipeline

    # Parse resolution
    try:
        w_str, h_str = args.resolution.split("x")
        width, height = int(w_str), int(h_str)
    except ValueError:
        print(f"[ERROR] Invalid resolution '{args.resolution}'. Use WxH, e.g. 128x128")
        sys.exit(1)

    dtype_str: str = args.dtype
    dtype = _dtype_str_to_torch(dtype_str)
    model_id: str = args.model_id
    num_steps: int = args.steps
    num_frames: int = args.frames
    warmup: int = args.warmup

    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device_type = device.type

    print(f"Benchmark: {width}x{height}, {num_frames} frames, {num_steps} steps, "
          f"dtype={dtype_str}, device={device_type}")

    timings: dict[str, float] = {}
    t_total_start = time.perf_counter()

    # --- Stage: model_load ---
    _sync(device_type)
    t0 = time.perf_counter()

    controlnet = ControlNetModel.from_pretrained(
        model_id, subfolder="controlnet", torch_dtype=dtype,
    )
    pipeline = StableVSRPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=dtype,
    )
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline.scheduler = scheduler
    pipeline = pipeline.to(device)

    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    # RAFT must stay float32 on MPS
    of_model = of_model.to(device, dtype=torch.float32)

    _sync(device_type)
    timings["model_load"] = time.perf_counter() - t0
    print(f"  model_load: {timings['model_load']:.2f}s")

    # --- Stage: pipeline_config ---
    _sync(device_type)
    t0 = time.perf_counter()

    pipeline.enable_model_cpu_offload()
    if device_type == "mps":
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_tiling()

    _sync(device_type)
    timings["pipeline_config"] = time.perf_counter() - t0
    print(f"  pipeline_config: {timings['pipeline_config']:.2f}s")

    # Synthetic frames
    rng = np.random.default_rng(42)
    frames = [
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
        for _ in range(num_frames)
    ]

    # Optional warmup iterations
    for wi in range(warmup):
        print(f"  warmup {wi + 1}/{warmup} ...")
        pipeline("", frames, num_inference_steps=num_steps, guidance_scale=0, of_model=of_model)
        _sync(device_type)

    # --- Stage: inference ---
    _sync(device_type)
    t0 = time.perf_counter()

    pipeline("", frames, num_inference_steps=num_steps, guidance_scale=0, of_model=of_model)

    _sync(device_type)
    timings["inference"] = time.perf_counter() - t0
    print(f"  inference: {timings['inference']:.2f}s")

    timings["total"] = time.perf_counter() - t_total_start

    peak_rss = _peak_rss_mb()

    # Human-readable report
    print("\n--- Benchmark Report ---")
    print(f"  Backend:    {device_type}")
    print(f"  Dtype:      {dtype_str}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames:     {num_frames}")
    print(f"  Steps:      {num_steps}")
    for stage, secs in timings.items():
        print(f"  {stage:20s} {secs:8.2f}s")
    if peak_rss is not None:
        print(f"  Peak RSS:   {peak_rss:.1f} MiB")

    # Optional JSON output
    if args.output:
        report = {
            "backend": device_type,
            "dtype": dtype_str,
            "resolution": f"{width}x{height}",
            "frames": num_frames,
            "steps": num_steps,
            "warmup": warmup,
            "timings": timings,
            "peak_rss_mib": peak_rss,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nJSON report saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="stablevsr",
        description="StableVSR: Video super-resolution with temporally-consistent diffusion models",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("backend-info", help="Show available compute backends")
    sub.add_parser("doctor", help="Run diagnostic checks")

    infer_parser = sub.add_parser("infer", help="Run video super-resolution inference")
    infer_parser.add_argument(
        "--input", required=True, help="Input folder of LR image sequences"
    )
    infer_parser.add_argument(
        "--output", required=True, help="Output folder for SR results"
    )
    infer_parser.add_argument(
        "--model-id", default="claudiom4sir/StableVSR", help="Model ID or local path"
    )
    infer_parser.add_argument(
        "--controlnet-ckpt", default=None, help="Custom controlnet checkpoint path"
    )
    infer_parser.add_argument(
        "--steps", type=int, default=50, help="Inference steps (default: 50)"
    )
    infer_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    infer_parser.add_argument(
        "--backend", default=None, help="Backend override (default: auto)"
    )
    infer_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load models in float16 (shortcut for --dtype float16)",
    )
    infer_parser.add_argument(
        "--dtype",
        default=None,
        choices=sorted(VALID_DTYPES),
        help="Model precision (default: float32). Restrictions per device are enforced.",
    )
    infer_parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable tiled VAE decode/encode (prevents OOM on large images)",
    )
    infer_parser.add_argument(
        "--vae-slicing",
        action="store_true",
        help="Enable sliced VAE decoding (reduces peak memory)",
    )
    infer_parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload idle models to CPU between forward passes",
    )
    infer_parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate args and backend selection without loading models",
    )
    infer_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging for backend selection",
    )

    # -- benchmark subcommand --
    bench_p = sub.add_parser("benchmark", help="Run a synthetic pipeline benchmark")
    bench_p.add_argument(
        "--steps", type=int, default=3, help="Inference steps (default: 3)",
    )
    bench_p.add_argument(
        "--frames", type=int, default=2, help="Number of synthetic frames (default: 2)",
    )
    bench_p.add_argument(
        "--resolution", default="128x128", help="Frame resolution WxH (default: 128x128)",
    )
    bench_p.add_argument(
        "--dtype", default="float32", choices=sorted(VALID_DTYPES),
        help="Model precision (default: float32)",
    )
    bench_p.add_argument(
        "--warmup", type=int, default=0, help="Warmup iterations before timed run (default: 0)",
    )
    bench_p.add_argument(
        "--output", default=None, help="Optional path for JSON report output",
    )
    bench_p.add_argument(
        "--model-id", default="claudiom4sir/StableVSR", help="Model ID or local path",
    )

    return parser


def main() -> None:
    """Entry point for the ``stablevsr`` CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "backend-info":
        cmd_backend_info(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
