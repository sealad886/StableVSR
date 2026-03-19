# Apple Silicon Guide

Running StableVSR on Apple Silicon Macs (M1/M2/M3/M4).

## TL;DR

**MLX backend (recommended):**

```bash
pip install -e ".[apple]"
python scripts/benchmark_mlx.py --scenarios smoke-160x90-1f-notile
```

**Torch-MPS backend (legacy):**

```bash
pip install -e ".[apple]"
stablevsr doctor
stablevsr infer --input ./frames --output ./sr --fp16 --vae-tiling
```

## Backends

### MLX (Primary)

The MLX backend provides native Apple Silicon inference with full Metal GPU
acceleration. All StableVSR components (UNet, VAE, ControlNet, scheduler) run
natively in MLX. Optical flow (RAFT) uses a minimal PyTorch bridge.

| Capability | Status |
|---|---|
| Inference | **Fully supported** |
| Training | Not supported |
| float16 | Recommended (default) |
| float32 | Works but slower |
| Tiled VAE | Auto-detected (>4096 latent px) |
| `mx.compile` | 2.15× UNet speedup |

See [benchmark_results.md](benchmark_results.md) for detailed performance data.

**Usage:**

```python
from stablevsr.mlx.pipeline import StableVSRMLXPipeline

pipe = StableVSRMLXPipeline(model_dir="models/StableVSR/models--claudiom4sir--StableVSR/snapshots/...")
results = pipe(
    prompt="",
    frames=frames,
    num_inference_steps=20,
    compile_models=True,      # JIT-compile UNet/ControlNet (recommended)
    ttg_start_step=5,         # Skip temporal guidance early steps
)
```

### Torch-MPS (Legacy)

| Capability | Status |
|---|---|
| Inference | Fully supported |
| Training | Not fully tested — use at own risk |
| float16 | Works (recommended) |
| bfloat16 | **Not supported** (MPS limitation) |
| float32 | Works but slower and uses more memory |

Auto-detection selects `torch-mps` on any Apple Silicon Mac. No manual override needed
unless you want to force CPU:

```bash
stablevsr infer --backend torch-cpu ...
```

## Installation

```bash
pip install -e ".[apple]"
```

This pulls PyTorch with MPS support and optional Apple-specific dependencies.
See [installation.md](installation.md) for full details.

## Verifying Your Setup

```bash
stablevsr doctor
```

The doctor command detects:

- **Architecture**: arm64 (native) vs x86_64 vs Rosetta 2
- **MPS availability**: whether `torch.backends.mps.is_available()` returns true
- **Backend selection**: confirms `torch-mps` is the auto-detected default

If doctor reports Rosetta 2, you're running an x86 Python. Reinstall with a native
arm64 Python (e.g. from python.org or Homebrew on arm64).

## Recommended Usage

```bash
stablevsr infer \
  --input ./lr_frames \
  --output ./sr_output \
  --fp16 \
  --vae-tiling
```

### Memory optimization flags

| Flag | Effect |
|---|---|
| `--vae-tiling` | Tiles VAE encode/decode. Strongly recommended on MPS to avoid OOM. |
| `--vae-slicing` | Slices VAE decoding to reduce peak memory. |
| `--cpu-offload` | Moves idle pipeline components to CPU between passes. Works on MPS. |

For 16 GB Macs, use `--fp16 --vae-tiling` at minimum. For 8 GB Macs, add
`--cpu-offload`.

## Dtype Rules

| dtype | MPS | Notes |
|---|---|---|
| float16 | yes | Default with `--fp16`. Best speed/memory tradeoff. |
| bfloat16 | **no** | MPS does not support bfloat16. Will error or silently fall back. |
| float32 | yes | Always works. ~2x slower, ~2x more memory than float16. |

Use `--fp16` unless you have a specific reason not to.

## Known MPS Limitations

1. **Silent CPU fallback** — Some PyTorch operations have no MPS kernel and silently
   execute on CPU. This works correctly but is slower than expected. Enable `--verbose`
   to see device placement warnings.

2. **Training ops** — MPS lacks kernels for some backward-pass operations. Training
   may fail or produce incorrect gradients. Training on MPS is not officially supported.

3. **bfloat16** — Not available on MPS. Do not pass `--dtype bfloat16`.

4. **Memory pressure** — macOS unified memory means GPU and system share the same pool.
   Large batches or high-resolution inputs can trigger system-level memory pressure.
   Use `--vae-tiling` and `--cpu-offload` proactively.

## MLX Optimization Flags

| Parameter | Effect | Recommendation |
|---|---|---|
| `compile_models=True` | JIT-compiles UNet/ControlNet for Metal shader fusion | Always use for >1 step |
| `ttg_start_step=N` | Skips temporal guidance on steps 0..N-1 | Use N ≤ num_steps // 2 |
| Tiled VAE | Auto-enabled for large latents (>4096 px area) | Automatic; configurable |

### Quality–Speed Tradeoff

- `compile_models=True`: No quality impact. ~1-2s JIT warmup on first call per shape.
- `ttg_start_step=1` with 2 steps: Minimal quality impact (TTG on final step only).
- `ttg_start_step=10` with 20 steps: Moderate quality risk — temporal coherence may
  degrade in early frames. Monitor inter-frame consistency visually.

## Why Not Rust/PyO3?

Profiling both backends shows **>99% GPU-bound** execution. The dominant
bottleneck (UNet, 69.6% of denoising) runs entirely as Metal GPU kernels.
`mx.compile()` provides JIT fusion that a Rust rewrite cannot improve upon.
See [ADR-0001](adr/0001-rust-acceleration-decision.md) for full analysis.

## Troubleshooting

### `stablevsr doctor` reports Rosetta 2

You're running an x86_64 Python binary under Rosetta. MPS will not be available.
Fix: install a native arm64 Python:

```bash
# Homebrew (arm64)
/opt/homebrew/bin/python3 -m venv .venv

# Or python.org universal installer — select arm64
```

### `MPS backend is not available`

- Requires macOS 12.3+ and PyTorch 1.12+.
- Verify: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Out of memory

Add `--vae-tiling --cpu-offload`. If still failing, drop to `--backend torch-cpu`
(slow but unbounded by GPU memory).

### Unexpectedly slow inference

Enable `--verbose` and check for CPU fallback warnings. Update PyTorch to the latest
stable release — MPS kernel coverage improves with each version.
