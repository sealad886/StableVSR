# Quality & Performance Tradeoffs

This document describes the optimization presets, their quality implications,
and how to choose settings for different use cases.

## Optimization Presets

StableVSR MLX inference supports four named presets, each representing a
different point on the quality–speed–memory curve.

| Preset | `mx.compile` | TTG Skip | Chunked | Description |
|---|---|---|---|---|
| **max-quality** | Off | None | Off | Reference quality, highest memory |
| **safe** | On | None | 16/4 | Good quality with bounded memory |
| **balanced** | On | 25% | 12/3 | Moderate tradeoff |
| **fast** | On | 50% | 8/2 | Maximum speed, quality risk |

### Preset Details

#### `max-quality`

- `compile_models=False`, `ttg_start_step=0`, no chunking
- All frames processed in a single pass with full temporal guidance
- Highest quality reference; no shortcuts applied
- Memory grows linearly with frame count — suitable for short clips only

#### `safe`

- `compile_models=True`, `ttg_start_step=0`, `chunk_size=16`, `chunk_overlap=4`
- JIT compilation for UNet/ControlNet (no quality impact)
- Full temporal texture guidance (no steps skipped)
- Bounded memory via chunking with overlap blending
- Recommended for production use with videos of any length

#### `balanced`

- `compile_models=True`, `ttg_start_step=25%`, `chunk_size=12`, `chunk_overlap=3`
- Skips temporal guidance on the first 25% of denoising steps
- Smaller chunks = lower peak memory, faster per-chunk
- Expected quality reduction in temporal consistency (not yet empirically quantified)

#### `fast`

- `compile_models=True`, `ttg_start_step=50%`, `chunk_size=8`, `chunk_overlap=2`
- Skips temporal guidance on the first 50% of denoising steps
- Aggressive chunking with smaller overlap
- Expected quality reduction; use `scripts/quality_compare.py` to evaluate on your content

## Optimization Mechanisms

### mx.compile (JIT Compilation)

Compiles UNet and ControlNet forward passes into fused GPU kernels.
Adds ~2s one-time compilation overhead, then speeds up each step.

**Quality impact:** None. Produces numerically identical results.

### Temporal Texture Guidance (TTG) Step Skip

The `ttg_start_step` parameter controls when inter-frame texture guidance begins.
Setting it to N means the first N denoising steps use no temporal guidance —
each frame is denoised independently for those steps.

**Quality impact:** Higher values reduce temporal consistency.
- `ttg_start_step=0`: Full guidance on every step (default, best quality)
- `ttg_start_step=steps//4`: Mild degradation, often imperceptible
- `ttg_start_step=steps//2`: Noticeable on scenes with subtle motion
- `ttg_start_step=steps`: Temporal guidance entirely disabled

### Temporal Chunking

Long videos are split into overlapping chunks. Each chunk runs the full
pipeline independently. Overlapping boundary frames are blended with a
linear cross-fade.

**Quality impact:**
- Chunk boundaries may show slight differences from single-pass processing
- Overlap blending reduces visible seams
- Larger `chunk_overlap` → smoother transitions, higher compute cost
- Larger `chunk_size` → better within-chunk temporal coherence

### Tiled VAE Decode

The VAE decoder can process large images in spatial tiles to avoid OOM.
Automatically enabled for resolutions above a threshold.

**Quality impact:** Minimal. Overlapping tiles with blending produce
results nearly identical to full-image decode.

## Memory Model

Approximate working-set memory per chunk (float16):

```
per_frame ≈ (4×H) × (4×W) × 3 × 2 bytes  (output pixels, f16)
           + (H) × (W) × 4 × 2 bytes      (latent, f16, vae_scale_factor=4)
```

For 480×270 input (→ 1920×1080 output):
- ~12 MB output pixels + ~4 MB latent per frame
- 16 frames ≈ ~260 MB working set
- Plus model weights (~5 GB)

## Guardrail System

The CLI and API check for risky parameter combinations before running:

| Code | Severity | Condition |
|---|---|---|
| `TTG_AGGRESSIVE` | warn | ttg_start_step > 50% of steps |
| `TTG_DISABLED` | warn | ttg_start_step >= num_steps |
| `MEMORY_HIGH` | warn | Estimated working set > 8 GB |
| `LONG_VIDEO_NO_CHUNK` | warn | >50 frames without chunking |
| `CHUNK_TOO_SMALL` | error | chunk_size < 2 |
| `OVERLAP_GE_CHUNK` | error | chunk_overlap >= chunk_size |
| `COMPILE_ONE_STEP` | warn | compile_models with 1 step |
| `NO_TILING_LARGE` | warn | Tiled VAE disabled for >1080p |

Warnings are logged; errors abort execution.

## Measuring Quality

Use the comparison script to evaluate presets side-by-side:

```bash
python scripts/quality_compare.py \
    --input e2e_test/input/frontdoor_clip \
    --output e2e_test/output/quality_compare \
    --steps 10
```

This reports:
- **PSNR** vs max-quality reference (higher = more similar)
- **Temporal stability** (lower = more consistent across frames)
- **Runtime** per frame

## Recommendations

| Use Case | Preset | Notes |
|---|---|---|
| Quality evaluation | `max-quality` | Short clips only (<20 frames) |
| Production processing | `safe` | Any length, good quality |
| Batch processing | `balanced` | Faster turnaround |
| Quick preview | `fast` | Rapid iteration |
| Custom | CLI flags | Fine-tune individual parameters |
