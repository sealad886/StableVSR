# Adversarial Validation Audit Report — StableVSR

**Date**: 2025-01-27  
**Scope**: Full codebase (17 files, 120+ functions)  
**Test baseline**: 38 tests → **57 tests** (all passing)  
**Commits**: `006a8b6` (batch 1), pending commit (batch 2)

---

## 1. Functional Inventory

| Component | Files | Role |
|---|---|---|
| Inference pipeline | `pipeline/stablevsr_pipeline.py` | SD 1.5 + ControlNet + TCM + RAFT bidirectional sampling |
| Training | `train.py` | HuggingFace Accelerate training loop |
| Evaluation | `eval.py` | PSNR, SSIM, LPIPS, DISTS, MUSIQ, NIQE, CLIP, tLPIPS, tOF |
| Testing script | `test.py` | Single-video / batch inference |
| Optical flow utils | `util/flow_utils.py` | Warp, flow computation, occlusion, gradients |
| DDPM scheduler | `scheduler/ddpm_scheduler.py` | Custom bidirectional DDPM |
| CLI | `src/stablevsr/cli.py` | `stablevsr` entry point (doctor, list-sequences, infer) |
| Backend system | `src/stablevsr/backends/` | Registry + torch/mlx backends |
| Dataset | `dataset/reds_dataset.py`, `dataset/config_reds.yaml` | REDS dataset loader |

---

## 2. Defects Found (22 Total)

### P0 — Critical (4 found, 4 fixed)

| ID | Location | Issue | Fix |
|---|---|---|---|
| P0-001 | `util/flow_utils.py:62` | Floor division `//` corrupted flow rescaling (integer truncation destroyed sub-pixel accuracy) | Changed to float division `/` |
| P0-002 | `train.py:156` | `image_logs` dict missing `validation_prompt` and `validation_image` keys → KeyError in logging | Added both keys |
| P0-003 | `train.py:176` | `wandb.Image()` called without importing wandb → NameError in W&B logging path | Added `import wandb` inside the branch |
| P0-004 | `train.py:227-228` | `save_model_card()` used undefined variables → crash on model save | Fixed argument passing |

### P1 — High (8 found, 8 fixed)

| ID | Location | Issue | Fix |
|---|---|---|---|
| P1-001 | `train.py:197` | `return image_logs` was indented inside the frame loop → returned after first frame only | Dedented to correct scope |
| P1-002 | `eval.py:13-14` | Deprecated RAFT API (`raft_large(pretrained=True)`) | Updated to `Raft_Large_Weights.DEFAULT` |
| P1-003 | `eval.py:85-88` | No existence check for GT directory → crash on missing sequences | Added `os.path.isdir()` guard with `continue` |
| P1-004 | `eval.py:157-161` | Empty temporal metric lists → `np.mean([])` produces NaN/warning | Added `if any(…) else 0.0` guard |
| P1-005 | `test.py:26` | Module-level `argparse` calls prevent import/testing | Wrapped in `if __name__ == "__main__":` |
| P1-006 | `eval.py:22` | Module-level `argparse` calls prevent import/testing | Wrapped in `if __name__ == "__main__":` |
| P1-007 | `test.py:5` | Frame loading accepted non-image files (`.DS_Store`, thumbs, etc.) | Added `IMAGE_EXTENSIONS` filter set |
| P1-008 | `registry.py:41+` | `select_backend("torch:mlx")` silently accepted invalid device suffixes | Added device suffix validation against `{"cpu", "cuda", "mps"}` |

### P2 — Medium (10 found, 8 fixed, 1 accepted, 1 deferred)

| ID | Location | Issue | Fix |
|---|---|---|---|
| P2-001 | `util/flow_utils.py:20` | `torch.meshgrid` without `indexing` kwarg → FutureWarning and ambiguity | Added `indexing="ij"` |
| P2-002 | `cli.py:116` | Return type annotation was `str` but function returns `tuple` | Fixed to `tuple` return type |
| P2-003 | `cli.py:235` | Version string hardcoded, diverged from `__version__` | Replaced with `__version__` import |
| P2-004 | `cli.py:108` | `list-sequences` showed hidden directories (`.DS_Store`, etc.) | Added hidden-entry filter |
| P2-005 | `mlx_backend.py:51` | `default_device()` returned `"gpu"` even when MLX unavailable | Returns `""` when `_MLX_AVAILABLE is False` |
| P2-006 | `torch_backend.py:19` | `is_available()` always returns `True` | **Accepted** — PyTorch is always importable; torch backend availability = torch importability |
| P2-007 | `test.py:14` | `center_crop` pads small images with black bands | **Deferred** — function is only used for optional crops, not in critical path |
| P2-008 | `registry.py` | MLX device suffix silently ignored | Resolved by P1-008 (device suffix validation) |
| P2-009 | `eval.py:157-161` | Empty sequence dicts produce NaN averages | Resolved by P1-004 (NaN guards) |
| P2-010 | `train.py:60` | `image_grid()` crashes on empty input list | Added early-return guard |

---

## 3. Functionality Validated as Correct

| Component | Validation Evidence |
|---|---|
| `StableVSRPipeline.__call__` | Code review: ControlNet conditioning, TCM temporal warping, bidirectional noise sampling all correctly wired. Flow warp uses corrected `get_flow`. |
| `DDPMScheduler` | Custom scheduler extends diffusers correctly. Previous/next frame noise scheduling is coherent. |
| `flow_warp()` | 5 unit tests (warp identity, shape, batch, error handling). Meshgrid indexing fixed. |
| `get_flow()` | 3 unit tests covering rescale_factor=1 and >1 paths. Division fix verified. |
| `get_flow_forward_backward()` | 1 adversarial test confirming both-direction flow computation. |
| `compute_flow_magnitude()` | Code review: straightforward squared magnitude, no issues. |
| `compute_flow_gradients()` | Code review: finite-difference gradients with correct boundary handling. |
| `detect_occlusion()` | Code review: forward-backward consistency check, threshold-based masking correct. |
| Backend registry | 19 unit tests covering detection, selection, fallback, device suffix validation, error paths. |
| CLI `doctor` | 3 tests: PyTorch reporting, no-backend error, CUDA/MPS reporting. |
| CLI `list-sequences` | 2 tests: directory collection with hidden-entry filtering. |
| CLI `infer` | Tested `--version` consistency with `__version__`. |
| Dataset loader | Code review: REDS metadata parsing, frame selection, crop logic all correct. |

---

## 4. Test Coverage Added

**19 new adversarial tests** across 3 test files (38 → 57):

### tests/test_backends.py (+7 tests)
- `TestRegistryDeviceSuffixValidation` (5): valid suffixes accepted, invalid rejected, error messages correct
- `TestMLXDefaultDevice` (2): empty string when unavailable, "gpu" when available

### tests/test_flow_utils.py (+4 tests)
- `TestGetFlow` (3): rescale=1, rescale>1, output shape
- `TestGetFlowForwardBackward` (1): both-direction flow
- `TestWarpError` (1): assertion error on mismatched shapes

### tests/test_cli.py (+5 tests)
- `TestCollectSequences` (2): directory collection with hidden-entry filtering
- `TestLoadFrames` (2): frame loading with image-extension filtering
- `TestVersionConsistency` (1): `__version__` matches CLI `--version`
- `TestImageExtensionFiltering` (1): IMAGE_EXTENSIONS constant + main guard verified
- `TestEvalMainGuard` (1): eval.py main guard verified

---

## 5. Docstrings Added

30 docstrings added across all modules in batch 1:

- `util/flow_utils.py`: All 7 functions documented (warp, flow, magnitude, gradients, occlusion, error)
- `pipeline/stablevsr_pipeline.py`: Pipeline class + `__call__` documented
- `scheduler/ddpm_scheduler.py`: Scheduler class + key methods documented
- `train.py`: `image_grid()`, `log_validation()`, `save_model_card()`, `parse_args()`, `main()` documented
- `eval.py`: Script-level purpose documented
- `test.py`: `center_crop()` documented
- `src/stablevsr/backends/`: All backend classes and methods documented (base, torch, mlx, registry)
- `src/stablevsr/cli.py`: All command handlers and helpers documented

---

## 6. Remaining Items

| Category | Item | Status |
|---|---|---|
| P2-006 | torch `is_available()` always True | Accepted by design — documented |
| P2-007 | `center_crop` pads small images | Deferred — non-critical helper, only used for optional preview crops |
| Integration test | End-to-end inference test with model weights | Blocked — requires ~6GB model download, not suitable for CI |
| GPU-specific tests | CUDA/MPS device-specific paths | Blocked — requires GPU hardware in CI |

---

## 7. Verification Evidence

```
$ python -m pytest tests/ -q
.........................................................  [100%]
57 passed, 2 warnings in 1.58s
```

All 57 tests pass with only third-party SWIG deprecation warnings (from pyiqa/DISTS dependencies, not our code).
