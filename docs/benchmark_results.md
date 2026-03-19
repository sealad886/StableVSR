# MLX StableVSR Benchmark Results

Collected on Apple Silicon (M-series, ~48 GB unified memory).
MLX 0.31.1, Python 3.11.15, float16 precision.

## Test Conditions

- **Input**: 480×270 LR frames from `e2e_test/input/frontdoor_clip/`
- **Output**: 1080×1920 SR frames (4× upscale)
- **Model**: `claudiom4sir/StableVSR` (local weights)
- **Optical flow**: RAFT (PyTorch CPU bridge)
- **VAE decode**: Tiled (auto-threshold 4096 latent px, tile_size=64, overlap=16)

## Summary: 2 Frames, 2 Steps, 480×270 → 1080×1920

| Configuration | Total (s) | UNet (s) | VAE decode (s) | ControlNet (s) | Peak Mem (GB) | Speedup |
|---|---|---|---|---|---|---|
| Baseline | 144.1 | 77.3 | 26.0 | 7.4 | 13.91 | 1.00× |
| `compile_models=True` | 101.7 | 36.0 | 24.5 | 9.5 | 13.91 | 1.42× |
| `ttg_start_step=1` | 89.9 | 37.8 | 12.6 | 8.0 | 13.91 | 1.60× |
| **Both combined** | **88.5** | **36.9** | **11.9** | **3.9** | **13.91** | **1.63×** |

## Stage Timing Breakdown (Baseline)

UNet inference dominates at **69.6%** of denoising loop time:

| Stage | Time (s) | % of Denoising |
|---|---|---|
| UNet (noise prediction) | 77.3 | 69.6% |
| VAE decode (temporal guidance) | 26.0 | 23.4% |
| ControlNet | 7.4 | 6.7% |
| Flow warp | 0.3 | 0.3% |
| Scheduler step | 0.1 | 0.1% |

## Optimization Details

### 1. `mx.compile()` for UNet/ControlNet

JIT-compiles model forward passes into fused GPU kernels, eliminating Python
dispatch overhead and enabling metal shader fusion.

- **UNet speedup: 2.15×** (77.3s → 36.0s)
- No memory increase
- Compile overhead amortized over loop iterations
- **Failure mode**: First call per input-shape configuration incurs ~1-2s JIT cost.
  For very short runs (1 step), compile overhead may exceed benefit.

### 2. Temporal Texture Guidance Step-Skip (`ttg_start_step`)

Skips expensive VAE decode + flow warp + ControlNet on early noisy denoising
steps where temporal guidance provides minimal benefit.

- **VAE decode calls halved**: 2 → 1 for 2-step run
- Also eliminates ControlNet forward passes for skipped steps
- **Failure mode**: Aggressive skipping (high `ttg_start_step` relative to total
  steps) may reduce inter-frame temporal consistency. For quality-critical work,
  keep `ttg_start_step ≤ num_steps // 2`.

## Smoke Scenarios (160×90, 1 frame, 2 steps)

| Configuration | Total (s) | Peak Mem (GB) | Output Shape |
|---|---|---|---|
| Non-tiled | 4.6 | 6.08 | (360, 640, 3) |
| Tiled (forced) | 5.8 | 5.06 | (360, 640, 3) |

Tiled decode uses 17% less memory at the cost of ~26% more time on small inputs
(due to tile overhead exceeding whole-decode cost).

## Scaling Estimates (Production Use)

For 5 frames × 20 steps at 480×270 (realistic production):

| Metric | Baseline (est.) | Optimized (est.) |
|---|---|---|
| UNet calls | 100 | 100 (2× faster each) |
| In-loop VAE decodes | 80 | 40 (skip first 10 steps) |
| Estimated total | ~55 min | ~28 min |

## Running Benchmarks

```bash
# Smoke scenarios (fast)
python scripts/benchmark_mlx.py --scenarios smoke-160x90-1f-notile smoke-160x90-1f-tiled

# Native resolution with optimizations
python scripts/benchmark_mlx.py --scenarios native-480x270-2f-compiled-ttg1

# Full matrix (warning: slow — native scenarios take 90-150s each)
python scripts/benchmark_mlx.py
```

Results are written to:
- `e2e_test/output/benchmark/results.json` (structured)
- `e2e_test/output/benchmark/benchmark_results.md` (human-readable)
