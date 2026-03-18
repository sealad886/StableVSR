"""StableVSR command-line interface."""

from __future__ import annotations

import argparse
import sys


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stablevsr",
        description="StableVSR: Video super-resolution with temporally-consistent diffusion models",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("backend-info", help="Show available compute backends")
    sub.add_parser("doctor", help="Run diagnostic checks")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "backend-info":
        cmd_backend_info(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
