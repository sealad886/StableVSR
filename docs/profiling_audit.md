# Profiling Audit: StableVSR on Apple Silicon

**Date:** 2025-03-18  
**System:** Apple M-series (MPS backend), 48 GB unified memory  
**PyTorch:** 2.10.0, diffusers 0.37.0, torchvision 0.25.0  
**Config:** 3 frames, 480×270 → 1920×1080, 5 diffusion steps, float32, CPU offload + attention slicing + VAE tiling  

## Executive Summary

VAE decode dominates inference at **64%** of pipeline time. The bidirectional
sampling strategy decodes the VAE **every step for every non-first frame** to
produce temporal texture guidance. This is the single largest optimization
target. UNet forward is second at 23%, RAFT optical flow and ControlNet are
~5.5% each. Flow warp and scheduler steps are negligible.

## Raw Measurements

### Top-Level Timings

| Stage | Seconds | % of Total |
|---|---:|---:|
| pipeline_call | 198.08 | 96.0% |
| load_controlnet | 3.48 | 1.7% |
| save_frames | 2.85 | 1.4% |
| import_core | 1.03 | 0.5% |
| load_pipeline | 0.76 | 0.4% |
| load_raft | 0.10 | 0.0% |
| backend_detect | 0.04 | 0.0% |
| frame_load | 0.01 | 0.0% |
| **total_wall** | **206.37** | **100%** |

### Method-Level Breakdown (Inside `pipeline.__call__`)

| Method | Mean (ms) | Calls | Total (s) | % of Pipeline |
|---|---:|---:|---:|---:|
| **vae_decode** | 9,798 | 13 | 127.38 | **64.3%** |
| **unet_forward** | 3,096 | 15 | 46.44 | **23.4%** |
| raft_get_flow | 2,773 | 4 | 11.09 | 5.6% |
| controlnet_forward | 1,093 | 10 | 10.93 | 5.5% |
| text_encoder | 968 | 1 | 0.97 | 0.5% |
| scheduler_step | 8 | 15 | 0.12 | 0.1% |
| flow_warp | 9 | 10 | 0.09 | <0.1% |

### Call Count Analysis

With 3 frames and 5 steps, bidirectional sampling means the pipeline processes
each timestep in alternating forward/reverse order across frames. This produces:

- **UNet calls:** 5 steps × 3 frames = 15
- **ControlNet calls:** 5 steps × (3−1) frames × direction = 10 (first frame skips ControlNet)
- **VAE decode (mid-loop):** 5 steps × (3−1) frames × direction = 10 (temporal guidance) + 3 (final decode) = 13
- **RAFT get_flow:** 2 flows × (3−1) pairs = 4 (pre-computed once)
- **Flow warp:** same pattern as ControlNet = 10

### Memory (Peak RSS)

| Checkpoint | MB |
|---|---:|
| Start | 16 |
| After frame load | 212 |
| After ControlNet load | 401 |
| After pipeline load | 450 |
| After RAFT load | 492 |
| After inference | 602 |

Note: CPU offloading keeps peak RSS low by moving models between CPU and MPS
device on demand. Without CPU offloading, all models in float32 exceed 47 GiB
MPS allocation and crash.

## Critical Finding: MPS float16 Incompatibility

Running with `--dtype float16` causes a Metal backend assertion failure:

```
MPSNDArrayMatrixMultiplication.mm:5028: failed assertion
'Destination NDArray and Accumulator NDArray cannot have different datatype'
```

This is a known PyTorch MPS limitation where certain matmul operations require
matching accumulator and destination dtypes. **float16 inference is currently
broken on MPS.** The pipeline must run in float32, which doubles memory usage
and approximately halves throughput compared to float16 on CUDA.

## Optimization Plan

Ranked by measured impact, implementation complexity, and risk.

| # | Bottleneck | Optimization | Expected Benefit | Complexity | Risk | Scope |
|---|---|---|---|---|---|---|
| 1 | VAE decode 64.3% | Reduce mid-loop VAE decodes: cache decoded x0 when latent unchanged, skip decode on first frame | **30-50% pipeline speedup** | Medium | Low | Python |
| 2 | VAE decode 64.3% | Smaller VAE tile size for MPS memory efficiency | Less OOM, enables more configs | Low | Low | Python |
| 3 | UNet 23.4% | torch.compile() with MPS backend (PyTorch 2.x) | 10-30% per-call speedup | Low | Medium (MPS compiler maturity) | Python |
| 4 | RAFT 5.6% | Pre-compute and cache optical flows to disk for repeated runs | Eliminates 11s on reruns | Low | None | Python |
| 5 | RAFT 5.6% | Run RAFT at reduced resolution (of_rescale_factor) | Proportional reduction | None (param exists) | Quality tradeoff | Config |
| 6 | ControlNet 5.5% | torch.compile() the ControlNet model | 10-20% per-call speedup | Low | Medium | Python |
| 7 | Model load 2.1% | Lazy model loading / keep in memory across frames | Eliminates reload overhead | Low | None | Python |
| 8 | Save 1.4% | Parallel frame saving with ThreadPoolExecutor | ~2-3× save speedup | Low | None | Python |
| 9 | flow_warp <0.1% | Cache meshgrid between calls | Negligible (8ms/call) | Trivial | None | Python |
| 10 | All models | Investigate MPS float16 fix / mixed precision | 2× memory + speed improvement | High | High (MPS bugs) | Python/System |

### Not Justified for Rust/PyO3

Based on measurements:
- **flow_warp** (8.9ms/call): Already fast, <0.1% of pipeline — Rust would save microseconds
- **scheduler_step** (8.2ms/call): Pure arithmetic, already fast in PyTorch
- **Preprocessing**: PIL/numpy operations are milliseconds total

The dominant bottlenecks (VAE decode, UNet, ControlNet) are neural network
forward passes that cannot be meaningfully accelerated by reimplementing in
Rust. They are bound by Metal GPU compute, not Python overhead.

## First Optimization Batch (Recommended)

### Opt-1: Reduce unnecessary VAE mid-loop decodes
The pipeline decodes x0_est through the VAE **every step for every non-first
frame** to produce warped temporal guidance. At 9.8s per decode, this is the
dominant cost. Opportunities:
- The first frame never uses temporal guidance → no ControlNet or VAE mid-decode needed
- If the scheduler produces identical x0_est at consecutive steps, the decode can be skipped

### Opt-2: Auto-enable safety defaults for MPS
Currently users must manually pass `--attention-slicing --vae-tiling` or the
pipeline crashes. The pipeline should auto-enable these when running on MPS.

### Opt-3: Benchmark CLI command
Add `stablevsr benchmark` that runs a standardized micro-benchmark and reports
per-stage timing, throughput, and memory usage.

### Opt-4: flow_warp meshgrid caching
`flow_warp()` creates `torch.meshgrid()` on every call. While timing is
negligible (8.9ms), this is a correctness-enabling optimization that also
reduces MPS memory allocator churn.
