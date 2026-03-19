# Quality Comparison Results

**Date:** 2025-03-19
**Hardware:** Apple Silicon, ~48 GB unified memory
**MLX:** 0.31.1, mlx-metal 0.31.1
**Input:** 3 frames at 480×270 RGB → 1920×1080 output
**Steps:** 10 denoising steps, seed=42
**Tool:** `scripts/quality_compare.py`

## Results

### Per-Preset Performance

| Preset | Compile | TTG Start | Time (s) | s/frame | Speedup vs ref | PSNR (dB) | Temporal Stability |
|---|---|---|---|---|---|---|---|
| max-quality | Off | 0 | 991.1 | 330.4 | 1.0× | ∞ (ref) | 0.291 |
| safe | On | 0 | 845.7 | 281.9 | 1.17× | 77.54 | 0.291 |
| balanced | On | 2 | 788.7 | 262.9 | 1.26× | 53.33 | 0.281 |
| fast | On | 5 | 414.3 | 138.1 | 2.39× | 43.70 | 0.304 |

### Stage Timing (seconds)

| Stage | max-quality | safe | balanced | fast |
|---|---|---|---|---|
| UNet | 527.6 (53%) | 449.3 (53%) | 425.1 (54%) | 209.1 (50%) |
| VAE decode | 230.4 (23%) | 231.1 (27%) | 189.1 (24%) | 108.9 (26%) |
| ControlNet | 175.6 (18%) | 112.1 (13%) | 101.4 (13%) | 25.6 (6%) |
| Flow warp | 8.0 (1%) | 3.8 (<1%) | 3.0 (<1%) | 1.4 (<1%) |
| Scheduler | 0.8 (<1%) | 0.3 (<1%) | 2.0 (<1%) | 0.1 (<1%) |

### PSNR Interpretation

- **>50 dB:** Virtually identical to reference
- **40–50 dB:** Excellent quality, differences imperceptible to human eye
- **30–40 dB:** Good quality, minor differences visible on close inspection
- **<30 dB:** Noticeable degradation

All presets produce output above 43 dB vs the max-quality reference, meaning quality
differences are imperceptible or near-imperceptible for practical use.

### Temporal Stability Interpretation

Temporal stability measures the standard deviation of inter-frame pixel differences.
Lower values indicate more consistent frame-to-frame behavior.

All presets fall in a tight range (0.281–0.304), indicating that TTG skip levels
used in these presets do not significantly impact temporal coherence.

## Analysis

### Compilation Impact (safe vs max-quality)

Both use `ttg_start_step=0`, isolating the effect of `mx.compile`:
- **14.7% wall-clock speedup** (330.4 → 281.9 s/frame)
- UNet: 14.8% faster (527.6 → 449.3s)
- ControlNet: 36.2% faster (175.6 → 112.1s)
- VAE decode: unchanged (not compiled)
- **PSNR: 77.54 dB** — safe is near-identical to max-quality
- **Temporal stability: identical** (0.291)
- Conclusion: `mx.compile` provides pure speedup with no quality impact

### TTG Skip Impact (balanced vs safe)

Both compiled, isolating `ttg_start_step=2` effect:
- **6.7% additional speedup** (281.9 → 262.9 s/frame)
- Fewer VAE decodes in early steps (231.1 → 189.1s)
- Fewer ControlNet calls (112.1 → 101.4s)
- **PSNR: 53.33 dB** — excellent quality, minor numerical differences
- **Temporal stability: slightly improved** (0.291 → 0.281)
- Conclusion: 25% TTG skip is a safe optimization with minimal quality impact

### Aggressive TTG Skip (fast vs max-quality)

Compile + 50% TTG skip:
- **2.39× total speedup** (330.4 → 138.1 s/frame)
- UNet: 60.3% faster (527.6 → 209.1s)
- ControlNet: 85.4% faster (175.6 → 25.6s) — most calls skipped
- VAE decode: 52.7% faster (230.4 → 108.9s) — fewer tiled decodes needed
- **PSNR: 43.70 dB** — still excellent, but largest quality drop
- **Temporal stability: slightly higher** (0.304 vs 0.291)
- Conclusion: fast preset offers major speedup; quality remains above perceptual threshold

## Limitations

- 3 frames only; chunking not activated (minimum chunk sizes are 8–16)
- Single test clip (`frontdoor_clip`)
- PSNR only; no perceptual metrics (LPIPS, FID)
- No SSIM computation (requires scikit-image)
