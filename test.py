import argparse
import os
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

import torch
from accelerate.utils import set_seed
from diffusers import ControlNetModel, DDPMScheduler
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from pipeline.stablevsr_pipeline import StableVSRPipeline


def center_crop(im, size=128):
    """Center-crop a PIL Image to the given square size."""
    width, height = im.size  # Get dimensions
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return im.crop((left, top, right, bottom))


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(description="Test code for StableVSR.")
    parser.add_argument(
        "--out_path",
        default="./StableVSR_results/",
        type=str,
        help="Path to output folder.",
    )
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to input folder (containing sets of LR images).",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of sampling steps"
    )
    parser.add_argument(
        "--controlnet_ckpt",
        type=str,
        default=None,
        help="Path to your folder with the controlnet checkpoint.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Load models in float16 (halves memory usage).",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable tiled VAE decode/encode (prevents OOM on large images).",
    )
    parser.add_argument(
        "--vae-slicing",
        action="store_true",
        help="Enable sliced VAE decoding (reduces peak memory).",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload idle models to CPU between forward passes.",
    )
    args = parser.parse_args()

    print("Run with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # set parameters
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model_id = "claudiom4sir/StableVSR"
    dtype = torch.float16 if args.fp16 else torch.float32
    controlnet_model = ControlNetModel.from_pretrained(
        args.controlnet_ckpt if args.controlnet_ckpt is not None else model_id,
        subfolder="controlnet",
        torch_dtype=dtype,
    )  # your own controlnet model
    pipeline = StableVSRPipeline.from_pretrained(
        model_id, controlnet=controlnet_model, torch_dtype=dtype
    )
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipeline.scheduler = scheduler
    pipeline = pipeline.to(device)
    if args.vae_tiling:
        pipeline.enable_vae_tiling()
        print("VAE tiling enabled")
    if args.vae_slicing:
        pipeline.enable_vae_slicing()
        print("VAE slicing enabled")
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
        print("Model CPU offload enabled")
    if device.type == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    of_model.requires_grad_(False)
    of_model = of_model.to(device)

    # iterate for every video sequence in the input folder
    seqs = sorted(os.listdir(args.in_path))
    for seq in seqs:
        frame_names = sorted(
            f
            for f in os.listdir(os.path.join(args.in_path, seq))
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        )
        frames = []
        for frame_name in frame_names:
            frame = Path(os.path.join(args.in_path, seq, frame_name))
            frame = Image.open(frame)
            # frame = center_crop(frame)
            frames.append(frame)

        # upscale frames using StableVSR
        frames = pipeline(
            "",
            frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=0,
            of_model=of_model,
        ).images
        frames = [frame[0] for frame in frames]

        # save upscaled sequences
        seq = Path(seq)
        target_path = os.path.join(args.out_path, seq.parent.name, seq.name)
        os.makedirs(target_path, exist_ok=True)
        for frame, name in zip(frames, frame_names):
            frame.save(os.path.join(target_path, name))
