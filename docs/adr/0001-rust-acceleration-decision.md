# ADR-0001: Rust/PyO3 Acceleration for StableVSR

**Status:** Rejected (reconfirmed)  
**Date:** 2025-03-18  
**Updated:** 2025-07-10 — reconfirmed with MLX backend profiling data  
**Context:** Phase 3 optimization pass on Apple Silicon inference  

## Decision

Do not introduce Rust/PyO3 acceleration components at this time. Profiling
evidence from both the torch-MPS and MLX backends shows that Python-layer
overhead is negligible compared to GPU-bound model forward passes.

## Context

StableVSR performs video super-resolution using Stable Diffusion 1.x +
ControlNet + RAFT optical flow with bidirectional temporal sampling. We
profiled the full inference pipeline on Apple Silicon (both MPS and MLX
backends) to identify whether any hot paths would benefit from a
compiled-language rewrite.

## Evidence (Profiling Results)

Configuration: 3 frames, 480×270 → 1920×1080, 5 steps, float32, MPS with CPU
offloading.

| Component | % of Pipeline | Mean per Call | Bottleneck Type |
|---|---:|---:|---|
| VAE decode | 64.3% | 9,798 ms | GPU compute (Metal) |
| UNet forward | 23.4% | 3,096 ms | GPU compute (Metal) |
| RAFT optical flow | 5.6% | 2,773 ms | GPU compute (Metal) |
| ControlNet forward | 5.5% | 1,093 ms | GPU compute (Metal) |
| Text encoder | 0.5% | 968 ms | GPU compute (Metal) |
| Scheduler step | 0.1% | 8 ms | CPU arithmetic |
| Flow warp | <0.1% | 9 ms | GPU grid_sample |

**97.8%** of pipeline time is spent in neural network forward passes that
execute as Metal GPU kernels. These cannot be meaningfully accelerated by
reimplementing surrounding Python in Rust.

The candidates for Rust acceleration and why they don't justify it:

- **flow_warp** (8.9 ms/call): Already uses `torch.grid_sample` which runs as
  a single Metal kernel. A Rust reimplementation would still need to call into
  Metal for the actual grid sampling. The Python overhead is ~microseconds.

- **scheduler_step** (8.2 ms/call): Pure tensor arithmetic that PyTorch JITs
  efficiently. Reimplementing in Rust+PyO3 would save low single-digit
  milliseconds at most, against a 198-second pipeline.

- **Preprocessing** (frame loading, resizing): PIL + numpy operations complete
  in tens of milliseconds total. Not a bottleneck.

- **Postprocessing** (save): I/O bound (disk write), not CPU bound.

## Consequences

### Positive
- No Rust toolchain dependency added to the project
- No PyO3/maturin build complexity for contributors
- No cross-platform compilation matrix to maintain
- Development effort focused on measured bottlenecks (VAE decode strategy, MPS defaults)

### Negative
- If a future Python-layer bottleneck emerges (e.g., custom video codec
  preprocessing at scale), this decision should be revisited with fresh
  profiling evidence.

### When to Revisit
- If profiling reveals a CPU-bound Python hot path consuming >5% of pipeline time
- If batch video processing introduces frame I/O as a bottleneck
- If a Rust-native optical flow library offers measurably faster inference than
  RAFT on the target hardware

## MLX Backend Profiling Update (2025-07-10)

The MLX native backend was profiled at 2 frames × 2 steps, 480×270 → 1080×1920,
float16 on Apple Silicon.

| Stage | Time (s) | % of Denoising | Bottleneck Type |
|---|---:|---:|---|
| UNet (noise prediction) | 77.3 | 69.6% | GPU compute (Metal) |
| VAE tiled decode (temporal guidance) | 26.0 | 23.4% | GPU compute (Metal) |
| ControlNet | 7.4 | 6.7% | GPU compute (Metal) |
| Flow warp | 0.3 | 0.3% | GPU interpolation |
| Scheduler step | 0.1 | 0.1% | CPU arithmetic |

**99.7%** GPU-bound on the MLX backend. The remaining 0.3% is flow warp (MLX
grid interpolation) and scheduler (CPU-side tensor arithmetic). Neither would
benefit from Rust reimplementation.

**Key improvement achieved via `mx.compile()`:** UNet 77.3s → 36.0s (2.15×
speedup) through JIT-compiled Metal shader fusion. This confirms that the
acceleration path for this workload is GPU kernel fusion, not compiled-language
Python replacement.

RAFT optical flow remains on PyTorch CPU. This is the only non-MLX component.
A Rust RAFT reimplementation would need to:
1. Reimplement the RAFT architecture in a native Metal compute pipeline
2. Match PyTorch's correlation volume computation precisely
3. Handle all flow normalization and rescaling

This exceeds reasonable ROI given RAFT accounts for <5% of pipeline time.
