# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — 2025-01-27

### Added

- **Optimization presets** (`max-quality`, `safe`, `balanced`, `fast`) with
  bundled compile, TTG, and chunking settings (`src/stablevsr/mlx/presets.py`)
- **Temporal chunking** for bounded-memory long-video inference with overlap
  blending and resume support (`src/stablevsr/mlx/chunked_pipeline.py`)
- **Guardrail system** that warns on risky parameter combinations and errors
  on invalid configurations before inference starts
- **`mlx-infer` CLI subcommand** with preset selection, chunk override flags,
  `--resume`, `--dry-run`, and per-frame output
- **`PipelineResult`** return type from MLX pipeline with `frames` and
  `stage_timing` fields
- **Quality comparison script** (`scripts/quality_compare.py`) for side-by-side
  preset evaluation with PSNR and temporal stability metrics
- **Quality tradeoffs documentation** (`docs/quality_tradeoffs.md`)
- Updated inference documentation with MLX CLI reference
- Test suites for presets (21 tests) and chunked pipeline (23 tests)

### Changed

- MLX pipeline `__call__` now returns `PipelineResult` instead of
  `list[np.ndarray]` — callers updated accordingly
- README updated to reflect full MLX backend support

## [0.1.0] — 2025-01-26

### Added

- Initial modernized codebase with backend registry
- MLX native inference pipeline (UNet, ControlNet, VAE, RAFT optical flow)
- `mx.compile` JIT compilation for UNet and ControlNet
- Temporal Texture Guidance step-skip optimization
- Tiled VAE decode for large resolutions
- Bidirectional frame traversal with flow warping
- CLI subcommands: `backend-info`, `doctor`, `infer`, `benchmark`
- Full test suite (149+ tests)
- Apple Silicon documentation and benchmark results
