"""End-to-end MLX pipeline test with real weights and test frames.

Runs 2 denoising steps on 2 frames at native resolution (480×270 → 1920×1080).
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("e2e_test")

MODEL_PATH = Path(
    "models/StableVSR/models--claudiom4sir--StableVSR/snapshots/"
    "fddd0e3921c22a5dcc6468c56c44abe6564bacc2"
)
FRAME_DIR = Path("e2e_test/input/frontdoor_clip")
OUTPUT_DIR = Path("e2e_test/output/mlx_test")


def main():
    import mlx.core as mx

    # 1. Load pipeline
    logger.info("Loading MLX pipeline from pretrained weights...")
    t0 = time.time()
    from stablevsr.mlx.pipeline import MLXStableVSRPipeline
    pipe = MLXStableVSRPipeline.from_pretrained(MODEL_PATH, dtype=mx.float16)
    t_load = time.time() - t0
    logger.info(f"Pipeline loaded in {t_load:.1f}s")

    # Log memory
    peak_mem = mx.metal.get_peak_memory() / (1024**3)
    logger.info(f"Peak GPU memory after load: {peak_mem:.2f} GB")

    # 2. Load test frames (just 2 for speed)
    frame_paths = sorted(FRAME_DIR.glob("*.png"))[:2]
    if len(frame_paths) < 2:
        logger.error(f"Need at least 2 frames in {FRAME_DIR}")
        sys.exit(1)

    frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
    logger.info(f"Loaded {len(frames)} frames: {frames[0].shape}")

    # 3. Load RAFT model
    logger.info("Loading RAFT optical flow model...")
    import torch
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    raft_model = raft_model.eval().cpu()
    for p in raft_model.parameters():
        p.requires_grad_(False)

    # 4. Run pipeline
    logger.info("Running pipeline (2 steps for smoke test)...")
    t0 = time.time()

    def progress(step, total):
        logger.info(f"  Step {step}/{total}")

    output_frames = pipe(
        images=frames,
        of_model=raft_model,
        prompt="clean, high-resolution, 8k, sharp, details",
        negative_prompt="blurry, noise, low-resolution, artifacts",
        num_inference_steps=2,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        seed=42,
        progress_callback=progress,
    )
    t_run = time.time() - t0
    logger.info(f"Pipeline done in {t_run:.1f}s")

    # 5. Validate outputs
    assert len(output_frames) == len(frames), f"Expected {len(frames)} frames, got {len(output_frames)}"
    for i, f in enumerate(output_frames):
        assert f.dtype == np.uint8, f"Frame {i}: wrong dtype {f.dtype}"
        assert f.ndim == 3 and f.shape[2] == 3, f"Frame {i}: wrong shape {f.shape}"
        # 4× upscale of 480×270 = 1920×1080
        logger.info(f"  Frame {i}: {f.shape}, pixel range=[{f.min()}, {f.max()}]")

    # 6. Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(output_frames):
        out_path = OUTPUT_DIR / f"frame_{i:04d}.png"
        Image.fromarray(f).save(out_path)
        logger.info(f"  Saved {out_path}")

    # 7. Memory stats
    peak_mem_final = mx.metal.get_peak_memory() / (1024**3)
    logger.info(f"Peak GPU memory: {peak_mem_final:.2f} GB")
    logger.info("E2E test PASSED")


if __name__ == "__main__":
    main()
