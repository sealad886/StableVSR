"""Bounded-memory long-video inference via temporal chunking.

Splits a long frame sequence into overlapping chunks, runs the base
MLXStableVSRPipeline on each chunk independently, and blends overlapping
boundary frames with a linear cross-fade.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkSpec:
    """One chunk in the execution plan."""

    chunk_index: int
    start_frame: int  # inclusive, global index
    end_frame: int  # exclusive, global index
    overlap_before: int  # frames shared with previous chunk
    overlap_after: int  # frames shared with next chunk

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame


def plan_chunks(
    num_frames: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkSpec]:
    """Compute a deterministic chunk plan for *num_frames*.

    Args:
        num_frames: Total frames in the video.
        chunk_size: Maximum frames per chunk.
        chunk_overlap: Number of frames shared between consecutive chunks.

    Returns:
        Ordered list of ChunkSpec. Covers all frames exactly once
        (except overlapping regions which appear in two adjacent chunks).

    Raises:
        ValueError: If parameters are invalid.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be >= 2, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    stride = chunk_size - chunk_overlap
    chunks: list[ChunkSpec] = []
    start = 0
    chunk_idx = 0

    while start < num_frames:
        end = min(start + chunk_size, num_frames)

        # If this chunk already covers all remaining frames, mark no overlap_after
        covers_end = (end == num_frames)

        # Minimum viable chunk: at least 2 frames (need >= 1 for pipeline)
        # If remainder is < 2 and we already have chunks, extend the last one
        remaining = end - start
        if remaining < 2 and chunks:
            chunks[-1] = ChunkSpec(
                chunk_index=chunks[-1].chunk_index,
                start_frame=chunks[-1].start_frame,
                end_frame=end,
                overlap_before=chunks[-1].overlap_before,
                overlap_after=0,
            )
            break

        overlap_before = chunk_overlap if chunk_idx > 0 else 0
        overlap_after = chunk_overlap if not covers_end else 0

        chunks.append(ChunkSpec(
            chunk_index=chunk_idx,
            start_frame=start,
            end_frame=end,
            overlap_before=overlap_before,
            overlap_after=overlap_after,
        ))

        if covers_end:
            break

        chunk_idx += 1
        start += stride

    return chunks


def blend_overlap(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    overlap: int,
) -> list[np.ndarray]:
    """Linearly blend *overlap* frames between two adjacent chunk outputs.

    Returns the blended frames (same length as *overlap*).
    frames_a must be the last *overlap* frames from chunk A.
    frames_b must be the first *overlap* frames from chunk B.
    """
    if overlap == 0:
        return []
    if len(frames_a) != overlap or len(frames_b) != overlap:
        raise ValueError(
            f"Expected {overlap} frames from each side, "
            f"got {len(frames_a)} and {len(frames_b)}"
        )

    blended = []
    for i in range(overlap):
        # Weight goes from 1.0 (favor A) at i=0 to 0.0 (favor B) at i=overlap-1
        # For overlap=1, alpha_a=0.5
        alpha_a = 1.0 - (i + 1) / (overlap + 1)
        alpha_b = 1.0 - alpha_a
        mixed = (
            frames_a[i].astype(np.float32) * alpha_a
            + frames_b[i].astype(np.float32) * alpha_b
        ).clip(0, 255).astype(np.uint8)
        blended.append(mixed)

    return blended


@dataclass
class ChunkManifest:
    """Tracks completed chunks for resume support."""

    total_frames: int
    chunk_size: int
    chunk_overlap: int
    completed_chunks: list[int] = field(default_factory=list)
    config_hash: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "total_frames": self.total_frames,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "completed_chunks": sorted(self.completed_chunks),
            "config_hash": self.config_hash,
        }, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "ChunkManifest":
        data = json.loads(path.read_text())
        return cls(
            total_frames=data["total_frames"],
            chunk_size=data["chunk_size"],
            chunk_overlap=data["chunk_overlap"],
            completed_chunks=data.get("completed_chunks", []),
            config_hash=data.get("config_hash", ""),
        )

    def is_chunk_done(self, chunk_index: int) -> bool:
        return chunk_index in self.completed_chunks

    def mark_done(self, chunk_index: int, path: Path) -> None:
        if chunk_index not in self.completed_chunks:
            self.completed_chunks.append(chunk_index)
        self.save(path)


def run_chunked_inference(
    pipeline: Any,
    images: list[np.ndarray],
    of_model: Any,
    *,
    chunk_size: int,
    chunk_overlap: int,
    output_dir: Path | None = None,
    resume: bool = False,
    dry_run: bool = False,
    prompt: str = "clean, high-resolution, 8k, sharp, details",
    negative_prompt: str = "blurry, noise, low-resolution, artifacts",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    seed: int | None = None,
    compile_models: bool = False,
    ttg_start_step: int = 0,
    force_tiled_vae: bool | None = None,
    progress_callback: Any | None = None,
) -> list[np.ndarray]:
    """Run the pipeline on overlapping temporal chunks with bounded memory.

    Args:
        pipeline: MLXStableVSRPipeline instance.
        images: All input frames as (H, W, 3) uint8 numpy arrays.
        of_model: PyTorch RAFT model.
        chunk_size: Frames per chunk.
        chunk_overlap: Overlap frames between adjacent chunks.
        output_dir: If set, save intermediate chunk outputs here.
        resume: If True and output_dir has a manifest, skip completed chunks.
        dry_run: If True, plan chunks and return empty list without running.
        prompt, negative_prompt, num_inference_steps, guidance_scale,
        controlnet_conditioning_scale, seed, compile_models, ttg_start_step,
        force_tiled_vae: Passed through to pipeline.__call__.
        progress_callback: Called with (chunk_idx, total_chunks, chunk_progress_msg).

    Returns:
        Final assembled list of SR frames in order.
    """
    import gc

    num_frames = len(images)
    chunks = plan_chunks(num_frames, chunk_size, chunk_overlap)

    # Log chunk plan
    logger.info(
        "Chunk plan: %d frames → %d chunks (size=%d, overlap=%d)",
        num_frames, len(chunks), chunk_size, chunk_overlap,
    )
    for c in chunks:
        logger.info(
            "  Chunk %d: frames [%d, %d) (%d frames, overlap_before=%d, overlap_after=%d)",
            c.chunk_index, c.start_frame, c.end_frame,
            c.num_frames, c.overlap_before, c.overlap_after,
        )

    if dry_run:
        logger.info("Dry run — returning empty result.")
        return []

    # Resume support
    manifest_path = output_dir / "chunk_manifest.json" if output_dir else None
    manifest: ChunkManifest | None = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume and manifest_path and manifest_path.exists():
            manifest = ChunkManifest.load(manifest_path)
            if (manifest.total_frames != num_frames
                    or manifest.chunk_size != chunk_size
                    or manifest.chunk_overlap != chunk_overlap):
                logger.warning(
                    "Manifest parameters mismatch — ignoring resume state."
                )
                manifest = None

    if manifest is None and manifest_path:
        manifest = ChunkManifest(
            total_frames=num_frames,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Run chunks
    chunk_outputs: dict[int, list[np.ndarray]] = {}
    total_time = 0.0

    for c in chunks:
        # Check resume
        if manifest and manifest.is_chunk_done(c.chunk_index):
            # Load from disk if available
            chunk_dir = output_dir / f"chunk_{c.chunk_index:04d}" if output_dir else None
            if chunk_dir and chunk_dir.exists():
                loaded = _load_chunk_frames(chunk_dir, c.num_frames)
                if loaded is not None:
                    chunk_outputs[c.chunk_index] = loaded
                    logger.info("Chunk %d: resumed from disk (%d frames)", c.chunk_index, len(loaded))
                    continue
            logger.warning("Chunk %d marked done but frames not found — rerunning.", c.chunk_index)

        logger.info(
            "Processing chunk %d/%d (frames %d-%d)...",
            c.chunk_index + 1, len(chunks), c.start_frame, c.end_frame - 1,
        )

        chunk_images = images[c.start_frame:c.end_frame]
        chunk_seed = seed + c.start_frame if seed is not None else None

        t0 = time.perf_counter()
        pipeline_result = pipeline(
            images=chunk_images,
            of_model=of_model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            seed=chunk_seed,
            compile_models=compile_models,
            ttg_start_step=ttg_start_step,
            force_tiled_vae=force_tiled_vae,
        )
        sr_frames = pipeline_result.frames
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        logger.info(
            "Chunk %d done in %.1fs (%d frames, %.1f s/frame)",
            c.chunk_index, elapsed, len(sr_frames), elapsed / max(len(sr_frames), 1),
        )

        chunk_outputs[c.chunk_index] = sr_frames

        # Save chunk to disk
        if output_dir:
            chunk_dir = output_dir / f"chunk_{c.chunk_index:04d}"
            _save_chunk_frames(sr_frames, chunk_dir, c.start_frame)

            if manifest and manifest_path:
                manifest.mark_done(c.chunk_index, manifest_path)

        if progress_callback:
            progress_callback(c.chunk_index + 1, len(chunks), f"chunk {c.chunk_index} done")

        # Free memory between chunks
        gc.collect()

    # Assemble final output with overlap blending
    logger.info("Assembling %d chunks with overlap blending...", len(chunks))
    final_frames = _assemble_chunks(chunks, chunk_outputs, chunk_overlap)

    logger.info(
        "Chunked inference complete: %d input frames → %d output frames in %.1fs",
        num_frames, len(final_frames), total_time,
    )

    # Save final manifest
    if output_dir and manifest_path:
        manifest_data = {
            "total_frames": num_frames,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "completed_chunks": sorted(manifest.completed_chunks) if manifest else [],
            "output_frames": len(final_frames),
            "total_time_s": round(total_time, 1),
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest_data, indent=2) + "\n"
        )

    return final_frames


def _assemble_chunks(
    chunks: list[ChunkSpec],
    chunk_outputs: dict[int, list[np.ndarray]],
    chunk_overlap: int,
) -> list[np.ndarray]:
    """Assemble chunk outputs into a single frame sequence with blended overlaps."""
    if not chunks:
        return []

    result: list[np.ndarray] = []

    for i, c in enumerate(chunks):
        frames = chunk_outputs[c.chunk_index]

        if i == 0:
            # First chunk: take all frames except the overlap_after region
            if c.overlap_after > 0 and i + 1 < len(chunks):
                result.extend(frames[:-c.overlap_after])
            else:
                result.extend(frames)
        else:
            # Non-first chunk: blend overlap_before, then add rest
            prev_chunk = chunks[i - 1]
            prev_frames = chunk_outputs[prev_chunk.chunk_index]

            if chunk_overlap > 0 and len(prev_frames) >= chunk_overlap:
                overlap_a = prev_frames[-chunk_overlap:]
                overlap_b = frames[:chunk_overlap]
                blended = blend_overlap(overlap_a, overlap_b, chunk_overlap)
                result.extend(blended)
                # Add remaining non-overlap frames
                remaining = frames[chunk_overlap:]
                if c.overlap_after > 0 and i + 1 < len(chunks):
                    result.extend(remaining[:-c.overlap_after])
                else:
                    result.extend(remaining)
            else:
                # No overlap to blend
                if c.overlap_after > 0 and i + 1 < len(chunks):
                    result.extend(frames[:-c.overlap_after])
                else:
                    result.extend(frames)

    return result


def _save_chunk_frames(
    frames: list[np.ndarray], chunk_dir: Path, start_idx: int
) -> None:
    """Save chunk frames as numbered PNGs."""
    from PIL import Image

    chunk_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(chunk_dir / f"frame_{start_idx + i:06d}.png")


def _load_chunk_frames(chunk_dir: Path, expected_count: int) -> list[np.ndarray] | None:
    """Load chunk frames from disk. Returns None if count doesn't match."""
    from PIL import Image

    paths = sorted(chunk_dir.glob("frame_*.png"))
    if len(paths) != expected_count:
        return None
    return [np.array(Image.open(p)) for p in paths]
