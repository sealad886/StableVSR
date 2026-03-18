"""StableVSR command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stablevsr import __version__


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


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run diagnostic checks."""
    issues: list[str] = []

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
        import imageio_ffmpeg  # noqa: F401

        print("[OK] imageio + ffmpeg available")
    except ImportError:
        print("[WARN] imageio/ffmpeg not installed (needed for video I/O)")

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


def cmd_infer(args: argparse.Namespace) -> None:
    """Run video super-resolution inference."""
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
    print(f"Using backend: {backend.name()} (device: {device})")

    # Seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load model
    model_id = args.model_id
    controlnet_src = args.controlnet_ckpt if args.controlnet_ckpt else model_id
    print(f"Loading controlnet from: {controlnet_src}")
    controlnet_model = ControlNetModel.from_pretrained(
        controlnet_src, subfolder="controlnet"
    )

    print(f"Loading pipeline from: {model_id}")
    pipeline = StableVSRPipeline.from_pretrained(model_id, controlnet=controlnet_model)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline.scheduler = scheduler
    pipeline = pipeline.to(device)

    # xformers — only when available and on CUDA
    if device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory-efficient attention")
        except Exception:
            pass

    # Optical flow model
    print("Loading RAFT optical-flow model")
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    of_model = of_model.to(device)

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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
