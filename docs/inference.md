# Inference Guide

Run video super-resolution with `stablevsr infer`.

## Quick Start

```bash
stablevsr infer --input ./lr_frames --output ./sr_output --fp16
```

This upscales every image sequence found under `./lr_frames` and writes results to `./sr_output`. Models are downloaded automatically from HuggingFace on first run.

## Input Format

The `--input` directory can be structured in two ways:

**Multiple sequences** — subdirectories, one per sequence:

```
lr_frames/
  clip_001/
    00000.png
    00001.png
    ...
  clip_002/
    00000.png
    00001.png
    ...
```

**Single sequence** — images directly in the folder:

```
lr_frames/
  frame_000.png
  frame_001.png
  ...
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`. Files are sorted lexicographically, so zero-padded numeric names keep correct order.

## CLI Reference

```
stablevsr infer [OPTIONS]
```

### Required

| Flag | Description |
|------|-------------|
| `--input PATH` | Directory containing LR image sequences (see Input Format above). |
| `--output PATH` | Directory where SR frames are written. Created automatically. |

### Model

| Flag | Default | Description |
|------|---------|-------------|
| `--model-id ID` | `claudiom4sir/StableVSR` | HuggingFace model ID or path to a local model directory. |
| `--controlnet-ckpt PATH` | *(none)* | Path to a custom ControlNet checkpoint. When omitted, the ControlNet bundled with `--model-id` is used. |

### Diffusion

| Flag | Default | Description |
|------|---------|-------------|
| `--steps N` | `50` | Number of DDPM diffusion steps. More steps = higher quality but slower. |
| `--seed N` | `42` | Random seed for reproducibility. |

### Precision & Backend

| Flag | Default | Description |
|------|---------|-------------|
| `--backend NAME` | `auto` | Force a specific compute backend (`torch-mps`, `torch-cuda`, `torch-cpu`). Auto-detection picks the best available. |
| `--fp16` | off | Shortcut for `--dtype float16`. |
| `--dtype DTYPE` | `float32` | Explicit precision: `float16`, `bfloat16`, or `float32`. Device restrictions are enforced at runtime (see table below). If both `--dtype` and `--fp16` are given, `--dtype` wins. |

**Dtype availability by device:**

| Device | float32 | float16 | bfloat16 |
|--------|---------|---------|----------|
| CPU    | yes     | —       | —        |
| CUDA   | yes     | yes     | yes      |
| MPS    | yes     | yes     | —        |

An unsupported dtype falls back to `float32` with a warning.

### Memory Optimization

| Flag | Description |
|------|-------------|
| `--vae-tiling` | Tile VAE encode/decode to avoid OOM on large resolutions. |
| `--vae-slicing` | Slice VAE decoding to reduce peak GPU memory. |
| `--cpu-offload` | Move idle pipeline components to CPU between forward passes. Trades latency for lower VRAM usage. |

On CUDA, xformers memory-efficient attention is enabled automatically when available.

### Debug

| Flag | Description |
|------|-------------|
| `--smoke-test` | Validate arguments and backend selection without loading any models. Useful for CI or checking a setup. |
| `--verbose` / `-v` | Enable debug-level logging. |

## Examples

### CUDA — default quality

```bash
stablevsr infer \
  --input ./REDS/test/lr \
  --output ./results/reds_sr \
  --fp16
```

### CUDA — low memory (8 GB VRAM)

```bash
stablevsr infer \
  --input ./lr_frames \
  --output ./sr_output \
  --fp16 --vae-tiling --cpu-offload
```

### Apple Silicon (MPS)

```bash
stablevsr infer \
  --input ./lr_frames \
  --output ./sr_output \
  --backend torch-mps --fp16
```

### CPU only

```bash
stablevsr infer \
  --input ./lr_frames \
  --output ./sr_output \
  --backend torch-cpu
```

CPU is float32-only and significantly slower. Use only when no GPU is available.

### Custom ControlNet checkpoint

```bash
stablevsr infer \
  --input ./lr_frames \
  --output ./sr_output \
  --controlnet-ckpt ./checkpoints/my_controlnet \
  --fp16
```

### Fewer steps (faster, lower quality)

```bash
stablevsr infer \
  --input ./lr_frames \
  --output ./sr_output \
  --steps 20 --fp16
```

### Dry run

```bash
stablevsr infer \
  --input ./lr_frames \
  --output ./sr_output \
  --fp16 --smoke-test
```

Prints resolved backend, device, dtype, and sequence count — loads nothing.

## Memory Tips

| Situation | Recommended flags |
|-----------|-------------------|
| 16+ GB VRAM | `--fp16` |
| 8–12 GB VRAM | `--fp16 --vae-tiling` |
| < 8 GB VRAM | `--fp16 --vae-tiling --vae-slicing --cpu-offload` |

---

## MLX Inference (Apple Silicon)

For native Apple Silicon inference without PyTorch, use the `mlx-infer` subcommand:

```bash
stablevsr mlx-infer --input ./lr_frames --output ./sr_output
```

### Preset-Based Usage

Presets bundle optimization settings for common use cases:

```bash
# Safe: bounded memory, full quality
stablevsr mlx-infer --input ./lr --output ./sr --preset safe

# Balanced: moderate speed/quality tradeoff
stablevsr mlx-infer --input ./lr --output ./sr --preset balanced

# Fast: maximum speed
stablevsr mlx-infer --input ./lr --output ./sr --preset fast

# Max quality: reference output, no chunking
stablevsr mlx-infer --input ./lr --output ./sr --preset max-quality --steps 50
```

See [Quality Tradeoffs](quality_tradeoffs.md) for detailed preset comparison.

### Long Video (Chunked Inference)

For videos exceeding available memory, chunked inference splits the sequence
into overlapping temporal chunks:

```bash
# Preset handles chunking automatically
stablevsr mlx-infer --input ./long_video --output ./sr --preset safe

# Or set chunk size explicitly
stablevsr mlx-infer --input ./long_video --output ./sr \
    --chunk-size 16 --chunk-overlap 4 --compile
```

Resume interrupted runs:
```bash
stablevsr mlx-infer --input ./long_video --output ./sr --preset safe --resume
```

Preview chunk plan without running:
```bash
stablevsr mlx-infer --input ./long_video --output ./sr --preset safe --dry-run
```

### MLX CLI Reference

```
stablevsr mlx-infer [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | *(required)* | Input directory of LR frames |
| `--output PATH` | *(required)* | Output directory for SR results |
| `--model-path PATH` | `models/StableVSR` | Local model cache path |
| `--preset NAME` | *(none)* | One of: `max-quality`, `safe`, `balanced`, `fast` |
| `--steps N` | `50` | Diffusion steps |
| `--seed N` | `42` | Random seed |
| `--compile` | *(off)* | Enable JIT compilation |
| `--no-compile` | *(off)* | Disable JIT compilation |
| `--ttg-start-step N` | `0` | TTG start step override |
| `--chunk-size N` | *(none)* | Frames per chunk |
| `--chunk-overlap N` | *(none)* | Overlap between chunks |
| `--resume` | *(off)* | Resume from manifest |
| `--dry-run` | *(off)* | Plan only, no inference |
| `--force-tiled-vae` | *(off)* | Force tiled VAE decode |
| `-v` / `--verbose` | *(off)* | Debug logging |
| Apple Silicon (unified memory) | `--backend torch-mps --fp16` |
| CPU-only machine | `--backend torch-cpu` (float32, slow) |

Stack `--vae-tiling`, `--vae-slicing`, and `--cpu-offload` freely — they are independent and additive.

## Other Subcommands

| Command | Description |
|---------|-------------|
| `stablevsr backend-info` | List detected compute backends and capabilities. |
| `stablevsr doctor` | Run diagnostic checks (PyTorch, diffusers, RAFT, ffmpeg). |
| `stablevsr --version` | Print the installed version. |
