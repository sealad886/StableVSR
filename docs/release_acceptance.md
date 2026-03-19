# Release Acceptance Test Report — v0.2.0

**Date:** 2025-03-19
**Platform:** Apple Silicon (macOS), ~48 GB unified memory
**MLX:** 0.31.1, mlx-metal 0.31.1
**Test input:** `e2e_test/input/frontdoor_clip/` — 5 frames at 480×270 RGB

## 1. Clean-Room Install (Python 3.11)

Fresh virtualenv `.venv_clean` created with Python 3.11.15 (Homebrew).

```
pip install -e ".[mlx]"
```

| Check | Result |
|---|---|
| `stablevsr --version` | 0.2.0 |
| `stablevsr backend-info` | MLX: AVAILABLE, torch-mps: AVAILABLE |
| `stablevsr doctor` | All checks passed |
| `stablevsr mlx-infer --help` | All flags shown |
| Smoke inference (5 frames, fast, 4 steps) | 333.2s — 5×1920×1080 PNGs |
| Chunked inference (5 frames, fast, chunk=3, overlap=1) | 495.1s — 5 frames |
| Dry-run mode | Correct — no inference executed |

## 2. Built-Artifact Verification

Build tool: Hatchling

| Artifact | Size | Status |
|---|---|---|
| `stablevsr-0.2.0-py3-none-any.whl` | 41 KB | OK |
| `stablevsr-0.2.0.tar.gz` | 126 KB | OK |

Build exclusions verified: `.venv*`, `models/`, `e2e_test/`, `.dolt/`, `.beads/`, `.codanna/`, `.fastembed_cache/`.

## 3. Wheel Install Test (Python 3.12)

Fresh virtualenv `.venv_wheel` created with Python 3.12.13.

```
pip install "dist/stablevsr-0.2.0-py3-none-any.whl[mlx]"
```

| Check | Result |
|---|---|
| `stablevsr --version` | 0.2.0 |
| `stablevsr --help` | All commands shown |
| `stablevsr backend-info` | MLX: AVAILABLE, torch-mps: AVAILABLE |
| `stablevsr mlx-infer --help` | All flags shown |

## 4. Quality Comparison

**Configuration:** 3 frames, 10 denoising steps, seed=42, 480×270 → 1920×1080

All presets ran successfully with `scripts/quality_compare.py`.

### Summary Table

| Preset | Time (s) | s/frame | Speedup | PSNR vs ref (dB) | Temporal Stability |
|---|---|---|---|---|---|
| max-quality | 991.1 | 330.4 | 1.0× | ∞ (reference) | 0.291 |
| safe | 845.7 | 281.9 | 1.17× | 77.54 | 0.291 |
| balanced | 788.7 | 262.9 | 1.26× | 53.33 | 0.281 |
| fast | 414.3 | 138.1 | 2.39× | 43.70 | 0.304 |

### Stage Timing Breakdown (seconds)

| Stage | max-quality | safe | balanced | fast |
|---|---|---|---|---|
| UNet | 527.6 | 449.3 | 425.1 | 209.1 |
| VAE decode | 230.4 | 231.1 | 189.1 | 108.9 |
| ControlNet | 175.6 | 112.1 | 101.4 | 25.6 |
| Flow warp | 8.0 | 3.8 | 3.0 | 1.4 |
| Scheduler | 0.8 | 0.3 | 2.0 | 0.1 |

### Key Findings

1. **`mx.compile` delivers 14.7% speedup** (safe vs max-quality, same TTG=0).
   UNet: 527.6→449.3s (-14.8%), ControlNet: 175.6→112.1s (-36.2%).
2. **TTG skip at 25% adds modest speedup** (balanced: 1.26× total).
   PSNR drops from 77.54→53.33 dB. Temporal stability slightly improves (0.291→0.281).
3. **TTG skip at 50% provides 2.39× speedup** (fast).
   PSNR drops to 43.70 dB. Temporal stability slightly degrades (0.291→0.304).
4. **All PSNR values are very high** — even 43.70 dB indicates nearly imperceptible differences.
   For reference: >40 dB is excellent; >30 dB is good quality.
5. **VAE decode is a significant bottleneck** (23% of max-quality time).
   Not affected by `mx.compile` (tiled decode is already optimized).
6. **Temporal stability is consistent** across all presets (0.281–0.304 range).

## 5. Release Decision

**GO** — v0.2.0 is ready for release.

### Evidence Summary

- Clean install works on Python 3.11 and 3.12
- Wheel and sdist artifacts are correctly sized and contain only source code
- All CLI entry points function correctly
- All 4 optimization presets produce valid 1920×1080 output
- Quality degradation from optimization is within acceptable bounds (PSNR >40 dB)
- Temporal stability is maintained across all presets
- Compilation provides measurable speedup with no quality impact
- TTG skip provides additional speedup with quantified quality tradeoff

### Known Limitations

- Chunking not testable with 3-frame clips (minimum chunk sizes are 8–16)
- Single test clip; broader content diversity not evaluated
- No automated perceptual quality metric (LPIPS/FID) — PSNR only
