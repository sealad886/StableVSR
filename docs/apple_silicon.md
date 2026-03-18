# Apple Silicon Guide

Running StableVSR on Apple Silicon Macs (M1/M2/M3/M4).

## TL;DR

```bash
pip install -e ".[apple]"
stablevsr doctor
stablevsr infer --input ./frames --output ./sr --fp16 --vae-tiling
```

## Backend: torch-mps

The `torch-mps` backend is the primary (and only functional) Apple Silicon backend.
It uses PyTorch's Metal Performance Shaders integration.

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

## Why Not MLX?

The `mlx` backend exists as a scaffold but **cannot run the StableVSR pipeline**.

| Blocker | Detail |
|---|---|
| RAFT optical flow | No MLX implementation exists. StableVSR depends on RAFT for temporal alignment. |
| ControlNet | Requires the `diffusers` library, which is PyTorch-native. |
| Temporal sampling | Bidirectional sampling logic is deeply coupled to PyTorch tensors. |
| Ecosystem maturity | MLX diffusion support is immature compared to `diffusers`. |

The MLX scaffold is retained for future exploration. Today, use `torch-mps`.

See [backends.md](backends.md) for the full backend capability matrix.

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
