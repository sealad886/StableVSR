#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import torch
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from diffusers import ControlNetModel, DDPMScheduler
from accelerate.utils import set_seed


LOG = logging.getLogger("stablevsr_macos")


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StableVSR on Apple Silicon (MPS) or CPU."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video file or directory of frames.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        default="output",
        help="Directory where output frames will be written.",
    )
    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional output video path (e.g. output/upscaled.mp4).",
    )
    parser.add_argument(
        "--model-dir",
        default="/Users/andrew/.cache/huggingface/hub/models--claudiom4sir--StableVSR/snapshots/fddd0e3921c22a5dcc6468c56c44abe6564bacc2",
        help="Local Hugging Face model snapshot directory downloaded via `hf download`.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the cloned StableVSR Git repo root. "
             "This script expects to import the local pipeline module from there.",
    )
    parser.add_argument(
        "--controlnet-dir",
        default=None,
        help="Optional override directory containing the ControlNet checkpoint folder. "
             "Defaults to ${MODEL_DIR}/controlnet.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Model dtype. float16 is the intended baseline on MPS.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=30,
        help="Diffusion sampling steps. Repo example uses 50; 30 is a more reasonable Mac default.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale. The repo test path uses 0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for output video when input is a frame directory or when input metadata is missing.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only process the first N frames. Useful for memory/smoke testing.",
    )
    parser.add_argument(
        "--attention-slicing",
        action="store_true",
        help="Enable attention slicing to reduce memory usage.",
    )
    parser.add_argument(
        "--vae-slicing",
        action="store_true",
        help="Enable VAE slicing to reduce memory usage.",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable VAE tiling to reduce memory usage on large frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output directory contents.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Optional[Path]]:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    controlnet_dir = Path(args.controlnet_dir).expanduser().resolve() if args.controlnet_dir else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}\n"
            "Download it first with:\n"
            "  hf download claudiom4sir/StableVSR --local-dir ./models/StableVSR"
        )

    if not repo_root.exists():
        raise FileNotFoundError(f"Repo root does not exist: {repo_root}")

    if not (repo_root / "pipeline" / "stablevsr_pipeline.py").exists():
        raise FileNotFoundError(
            f"Did not find repo pipeline code at: {repo_root / 'pipeline' / 'stablevsr_pipeline.py'}\n"
            "Clone the source repo and point --repo-root at it."
        )

    if output_dir.exists():
        if any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}\n"
                "Pass --overwrite if you want to reuse it."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    return input_path, output_dir, model_dir, controlnet_dir


def add_repo_to_syspath(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was explicitly requested but is not available.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float16":
        if device.type == "cpu":
            LOG.warning("float16 on CPU is slow and often a bad idea; forcing float32.")
            return torch.float32
        return torch.float16
    return torch.float32


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def load_frames_from_video(path: Path, max_frames: Optional[int]) -> Tuple[List[Image.Image], Optional[float]]:
    LOG.info("Reading video: %s", path)
    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = meta.get("fps")
    frames: List[Image.Image] = []

    try:
        for idx, frame_np in enumerate(reader):
            if max_frames is not None and idx >= max_frames:
                break
            frames.append(Image.fromarray(frame_np).convert("RGB"))
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")

    LOG.info("Decoded %d frames from video.", len(frames))
    return frames, fps


def load_frames_from_directory(path: Path, max_frames: Optional[int]) -> Tuple[List[Image.Image], List[Path]]:
    LOG.info("Reading frame directory: %s", path)
    files = sorted([p for p in path.iterdir() if is_image_file(p)])
    if not files:
        raise RuntimeError(f"No image frames found in directory: {path}")

    if max_frames is not None:
        files = files[:max_frames]

    frames = [Image.open(p).convert("RGB") for p in files]
    LOG.info("Loaded %d input frames.", len(frames))
    return frames, files


def default_output_video_path(output_dir: Path) -> Path:
    return output_dir / "stablevsr_output.mp4"


def write_output_frames(frames: List[Image.Image], output_dir: Path) -> List[Path]:
    output_paths: List[Path] = []
    for idx, frame in enumerate(frames):
        out_path = output_dir / f"{idx:06d}.png"
        frame.save(out_path)
        output_paths.append(out_path)
    LOG.info("Wrote %d output frames to %s", len(output_paths), output_dir)
    return output_paths


def write_output_video(frames: List[Image.Image], output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Encoding output video: %s (fps=%s)", output_path, fps)
    with imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    ) as writer:
        for frame in frames:
            writer.append_data(imageio.asarray(frame))
    LOG.info("Wrote output video: %s", output_path)


def load_pipeline(
    repo_root: Path,
    model_dir: Path,
    controlnet_dir: Optional[Path],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
):
    add_repo_to_syspath(repo_root)

    # Local import after sys.path modification.
    from pipeline.stablevsr_pipeline import StableVSRPipeline

    local_files_only = True
    controlnet_source = controlnet_dir if controlnet_dir is not None else model_dir

    LOG.info("Loading ControlNet from %s", controlnet_source)
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_source),
        subfolder="controlnet",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )

    LOG.info("Loading StableVSR pipeline from %s", model_dir)
    pipe = StableVSRPipeline.from_pretrained(
        str(model_dir),
        controlnet=controlnet,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        safety_checker=None,
        requires_safety_checker=False,
    )

    LOG.info("Loading scheduler from %s", model_dir)
    scheduler = DDPMScheduler.from_pretrained(
        str(model_dir),
        subfolder="scheduler",
        local_files_only=local_files_only,
    )
    pipe.scheduler = scheduler

    if args.attention_slicing:
        LOG.info("Enabling attention slicing.")
        pipe.enable_attention_slicing()

    if args.vae_slicing:
        LOG.info("Enabling VAE slicing.")
        pipe.enable_vae_slicing()

    if args.vae_tiling:
        LOG.info("Enabling VAE tiling.")
        pipe.enable_vae_tiling()

    LOG.info("Moving pipeline to device=%s dtype=%s", device, dtype)
    pipe = pipe.to(device)

    return pipe


def load_optical_flow_model(device: torch.device):
    LOG.info("Loading RAFT optical-flow model.")
    # Keep RAFT in default dtype; do not force half precision here.
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
    of_model.requires_grad_(False)
    of_model.eval()
    of_model = of_model.to(device)
    return of_model


def maybe_empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    input_path, output_dir, model_dir, controlnet_dir = validate_paths(args)
    repo_root = Path(args.repo_root).expanduser().resolve()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    LOG.info("Selected device: %s", device)
    LOG.info("Selected dtype: %s", dtype)

    if device.type == "mps" and dtype != torch.float16 and dtype != torch.float32:
        raise RuntimeError("Only float16/float32 are supported in this script.")

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    if is_video_file(input_path):
        input_frames, detected_fps = load_frames_from_video(input_path, args.max_frames)
        input_frame_paths = None
    elif input_path.is_dir():
        input_frames, input_frame_paths = load_frames_from_directory(input_path, args.max_frames)
        detected_fps = None
    else:
        raise RuntimeError(
            f"Unsupported input path: {input_path}\n"
            "Use a video file or a directory containing image frames."
        )

    fps = args.fps if args.fps is not None else (detected_fps if detected_fps else 24.0)
    if fps <= 0:
        raise RuntimeError(f"Invalid fps: {fps}")

    pipe = load_pipeline(
        repo_root=repo_root,
        model_dir=model_dir,
        controlnet_dir=controlnet_dir,
        device=device,
        dtype=dtype,
        args=args,
    )
    of_model = load_optical_flow_model(device)

    LOG.info(
        "Running inference on %d frames with num_inference_steps=%d guidance_scale=%.3f",
        len(input_frames),
        args.num_inference_steps,
        args.guidance_scale,
    )

    try:
        with torch.inference_mode():
            result = pipe(
                "",
                input_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                of_model=of_model,
            )
    except RuntimeError as exc:
        maybe_empty_cache(device)
        raise RuntimeError(
            "StableVSR inference failed. "
            "On Mac, first reduce --max-frames, lower --num-inference-steps, and enable "
            "--attention-slicing --vae-slicing --vae-tiling."
        ) from exc

    out_frames = result.images
    if not isinstance(out_frames, list) or not out_frames:
        raise RuntimeError("Pipeline returned no frames.")

    # Repo test code unwraps each item as frame[0].
    normalized_frames: List[Image.Image] = []
    for idx, item in enumerate(out_frames):
        if isinstance(item, list):
            if not item:
                raise RuntimeError(f"Empty frame list returned at index {idx}.")
            frame = item[0]
        else:
            frame = item

        if not isinstance(frame, Image.Image):
            raise RuntimeError(
                f"Unexpected output type at frame {idx}: {type(frame)!r}. "
                "Expected PIL.Image.Image or a single-item list containing it."
            )
        normalized_frames.append(frame)

    write_output_frames(normalized_frames, output_dir)

    output_video_path: Optional[Path] = None
    if args.output_video:
        output_video_path = Path(args.output_video).expanduser().resolve()
    elif is_video_file(input_path):
        output_video_path = default_output_video_path(output_dir)

    if output_video_path is not None:
        write_output_video(normalized_frames, output_video_path, fps)

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOG.error("%s: %s", exc.__class__.__name__, exc)
        raise
