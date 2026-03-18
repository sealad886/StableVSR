# StableVSR Modernization Audit

**Date**: 2026-03-18
**Auditor**: Automated engineering audit
**Repo**: claudiom4sir/StableVSR (ECCV 2024 paper)

---

## 1. Repository Structure

```
StableVSR/
├── pipeline/stablevsr_pipeline.py   # Core diffusion pipeline (~1100 lines)
├── scheduler/ddpm_scheduler.py      # Custom DDPM scheduler (~400 lines)
├── util/flow_utils.py               # Optical flow warping utilities
├── dataset/                         # REDS dataset loader + config
│   ├── reds_dataset.py              # Based on basicsr
│   ├── config_reds.yaml             # Hardcoded /home/crota/ paths
│   └── REDS_train_metadata.txt
├── test.py                          # Inference entrypoint (CUDA-only)
├── train.py                         # Training script (accelerate-based, CUDA-only)
├── eval.py                          # Evaluation metrics script (CUDA-only)
├── train.sh                         # Multi-GPU training launcher
├── run_stablevsr_mac.py             # macOS inference wrapper (added locally)
├── requirements.txt                 # Main deps (CUDA-centric)
├── requirements-mac.txt             # macOS inference subset (added locally)
├── models/StableVSR/               # Local HuggingFace model cache
├── README.md                        # Upstream docs
└── LICENSE                          # License file
```

## 2. Current Inference Entrypoints

### test.py (upstream, CUDA-only)
- Hardcodes `device = torch.device('cuda')`
- Calls `pipeline.enable_xformers_memory_efficient_attention()` (xformers is CUDA-only)
- Loads model from HuggingFace Hub via `from_pretrained('claudiom4sir/StableVSR')`
- No dtype control, no memory optimization options
- Input: directory of frame directories `in_path/sequence/frames`
- Output: upscaled frames to `out_path/sequence/frames`

### run_stablevsr_mac.py (local, MPS/CPU)
- Well-structured argparse CLI with device/dtype selection
- Supports video file or frame directory input
- MPS + CPU device selection (auto-detect)
- Memory optimization flags: `--attention-slicing`, `--vae-slicing`, `--vae-tiling`
- Loads from local model directory (not Hub)
- Video I/O via imageio/ffmpeg
- Good error handling and logging
- **This is the best starting point for the modernized inference CLI**

## 3. Training Entrypoint

### train.py
- Full HuggingFace accelerate-based training script
- CUDA-assumed throughout (xformers, GPU offloading)
- Uses custom DDPMScheduler from `scheduler/ddpm_scheduler.py`
- Dataset: REDSRecurrentDataset (depends on `basicsr` library)
- Configuration via argparse + YAML
- Validation uses `wandb` or `tensorboard`
- NOT a target for Phase 1-4 modernization

### train.sh
- Multi-GPU launcher with hardcoded CUDA_VISIBLE_DEVICES
- Hardcoded paths to `/home/crota/` datasets

## 4. Evaluation Entrypoint

### eval.py
- CUDA-only (`device = torch.device('cuda')`)
- Heavy metric dependencies: torchmetrics, pyiqa, DISTS_pytorch
- Computes: PSNR, SSIM, LPIPS, DISTS, MUSIQ, NIQE, CLIPIQA, tLPIPS, tOF
- Uses RAFT optical flow for temporal metrics
- NOT a target for Phase 1-4

## 5. Core Pipeline Architecture

### pipeline/stablevsr_pipeline.py
The pipeline is a ControlNet-based Stable Diffusion pipeline for video super-resolution:

**Architecture**: SD 1.x base + ControlNet (Temporal Conditioning Module)
- **VAE**: AutoencoderKL (encode/decode latents)
- **Text Encoder**: CLIPTextModel (empty prompt "" typical)
- **UNet**: UNet2DConditionModel (9-channel input: 4 latent + 4 LR concat + 1 noise)
- **ControlNet**: Takes warped previous frame estimate as conditioning
- **Scheduler**: Custom DDPM from `scheduler/ddpm_scheduler.py`
- **Optical Flow**: RAFT large model for temporal alignment

**Inference Flow** (per video):
1. Preprocess frames → bicubic 4x upscale
2. Compute forward/backward optical flows via RAFT
3. For each timestep, for each frame (bidirectional):
   - Decode previous frame's x0 estimate
   - Warp to current frame via optical flow
   - ControlNet conditioning with warped estimate
   - UNet denoising step
   - Reverse frame order each timestep (bidirectional sampling)
4. VAE decode final latents → output images

**Key Observations**:
- Inherits from `DiffusionPipeline` (diffusers)
- Uses `from_pretrained()` pattern for model loading
- Pipeline output: list of lists (each frame returns [pil_image])
- Heavy memory: loads VAE + UNet + ControlNet + CLIP + RAFT simultaneously

### scheduler/ddpm_scheduler.py
- Custom DDPMScheduler extending diffusers SchedulerMixin
- Returns `pred_original_sample` (x0 estimate) needed by TCM
- Standard diffusers config/registration pattern

### util/flow_utils.py
- `flow_warp()`: grid_sample-based optical flow warping
- `get_flow()`: RAFT inference wrapper
- `detect_occlusion()`: Forward-backward flow consistency
- `compute_flow_gradients()`: **Hardcodes `.to('cuda')`** ← breaking on non-CUDA
- `detect_occlusion()`: **Hardcodes `.to('cuda')`** ← breaking on non-CUDA

## 6. Where PyTorch Is Assumed Directly

| Location | Issue |
|---|---|
| `test.py:34` | `device = torch.device('cuda')` hardcoded |
| `eval.py:24` | `device = torch.device('cuda')` hardcoded |
| `util/flow_utils.py:61,86` | `.to('cuda')` hardcoded in `compute_flow_gradients()` and `detect_occlusion()` |
| `pipeline/stablevsr_pipeline.py:231` | `enable_model_cpu_offload()` hardcodes `torch.device(f"cuda:{gpu_id}")` |
| `pipeline/stablevsr_pipeline.py:1043` | `torch.cuda.empty_cache()` in offload path |
| `train.py` | CUDA-assumed throughout |

## 7. Where Diffusers-Specific Assumptions Exist

- Pipeline inherits `DiffusionPipeline`, `TextualInversionLoaderMixin`, `LoraLoaderMixin`, `FromSingleFileMixin`
- Model loading uses `from_pretrained()` with HuggingFace Hub structure
- Scheduler uses diffusers `ConfigMixin`, `register_to_config`
- `VaeImageProcessor` for image pre/post-processing
- `randn_tensor` from `diffusers.utils.torch_utils`
- Minimum diffusers version: `0.21.0.dev0`

**Impact**: The entire model pipeline is deeply coupled to PyTorch + diffusers. MLX cannot replace this without either:
(a) Full reimplementation of diffusers pipeline abstractions in MLX, or
(b) Weight conversion and custom MLX inference loop

## 8. CUDA-Only Dependencies

| Package | Version | Purpose | macOS Compatible? |
|---|---|---|---|
| `xformers` | 0.0.21 | Memory-efficient attention | **No** (CUDA-only) |
| `basicsr` | 1.4.2 | Dataset utilities for training | Partial (has CUDA extensions) |
| `bitsandbytes` | 0.43.3 | 8-bit Adam optimizer | **No** (CUDA-only) |
| `DISTS_pytorch` | 0.1 | Perceptual metric | Yes (pure torch) |
| `pyiqa` | 0.1.7 | Image quality metrics | Yes (pure torch) |
| `einops` | 0.8.0 | Tensor reshaping | Yes |

## 9. Dependency Graph (Inference Path)

**Required for inference**:
- `torch` ≥2.0 (MPS support)
- `torchvision` (RAFT optical flow)
- `diffusers` ≥0.21 (pipeline, VAE, UNet, ControlNet)
- `transformers` (CLIPTextModel, CLIPTokenizer)
- `accelerate` (set_seed, model loading utilities)
- `safetensors` (weight loading)
- `numpy`, `Pillow` (image I/O)
- `imageio`, `imageio-ffmpeg` (video I/O - used by run_stablevsr_mac.py)

**NOT required for inference**:
- `xformers` (CUDA optimization only)
- `basicsr` (training dataset only)
- `bitsandbytes` (8-bit training optimizer)
- `omegaconf` (training config)
- `einops` (training only)
- torchmetrics, pyiqa, DISTS_pytorch (evaluation only)

## 10. Pinned Versions Analysis

Current `requirements.txt` pins are very old:
- `torch==2.0.1` (Jun 2023) → current stable: 2.6.x
- `diffusers==0.21.1` (Sep 2023) → current stable: 0.32.x
- `transformers==4.33.1` (Sep 2023) → current stable: 4.48.x
- `accelerate==0.23.0` (Sep 2023) → current stable: 1.3.x
- `numpy==1.24.4` → current stable: 2.2.x (1.x deprecated)

**Recommendation**: Upgrade inference deps to current stable versions. Training deps stay at compatible versions until Phase 5.

## 11. MLX Feasibility Assessment

### Fully Portable Now
- **Preprocessing**: Bicubic upscaling, image normalization → trivial in MLX
- **Postprocessing**: Image denormalization, PIL conversion → trivial
- **Flow warping**: Grid sample is available in MLX (`mx.nn.functional` or manual)

### Partially Portable (Hybrid Path)
- **VAE encode/decode**: The SD VAE architecture (conv + attn) has MLX equivalents in `mlx-community` repos. Weight conversion from diffusers safetensors is documented.
- **Text encoder**: CLIP text encoder has MLX implementations (mlx-community/clip).
- **Scheduler math**: Pure arithmetic on tensors → portable to MLX

### Blocked / Very Difficult
- **UNet2DConditionModel**: The specific 9-channel input variant with ControlNet conditioning requires careful architecture matching. Standard mlx-community SD implementations use 4-channel UNet. The concatenated LR input channel adds complexity.
- **ControlNet**: No standard MLX ControlNet implementation exists. Would require custom porting.
- **RAFT optical flow**: torchvision RAFT has no MLX equivalent. Would need to stay on PyTorch/MPS.
- **DiffusionPipeline orchestration**: The bidirectional frame-wise sampling loop, temporal texture guidance, and flow warping integration are custom to this paper and tightly coupled to PyTorch tensors.

### Verdict: Option C (Torch-MPS primary, MLX future-ready)

The core model architecture (9-channel UNet + ControlNet + RAFT + bidirectional sampling) is fundamentally coupled to PyTorch/diffusers and cannot be cleanly ported to MLX without:
1. Re-implementing the custom UNet variant in MLX
2. Implementing ControlNet in MLX
3. Converting all model weights
4. Re-implementing the entire inference pipeline

This is a multi-week engineering effort with significant correctness risk. The correct approach is:

- **Primary Apple backend**: `torch-mps` (works today with the existing pipeline)
- **MLX backend**: Scaffold with honest capability reporting; implement conversion experiments
- **Future**: When mlx-community ships ControlNet support and custom UNet loading, revisit

## 12. Smallest Inference Path to Stabilize First

```
run_stablevsr_mac.py
  → pipeline/stablevsr_pipeline.py (StableVSRPipeline)
  → scheduler/ddpm_scheduler.py (DDPMScheduler)
  → util/flow_utils.py (flow_warp, get_flow)
  → torchvision RAFT model
  → diffusers ControlNet, UNet, VAE, CLIP
```

This is already working on MPS via `run_stablevsr_mac.py`. The stabilization work is:
1. Fix hardcoded `.to('cuda')` in `util/flow_utils.py`
2. Package as proper CLI behind backend abstraction
3. Add proper error handling and backend reporting

## 13. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Dependency upgrades break pipeline behavior | Medium | High | Pin tested versions, smoke test after each upgrade |
| R2 | MPS float16 produces different results than CUDA | Medium | Medium | Document precision differences, allow float32 fallback |
| R3 | MLX porting expectations exceed feasibility | High | High | Implement Option C honestly, document blockers |
| R4 | RAFT optical flow OOM on large frames via MPS | Medium | Medium | Document memory limits, provide --max-frames |
| R5 | basicsr dependency breaks macOS install | High | Low | Isolate to training extras only |
| R6 | diffusers API changes between 0.21→0.32 | Medium | High | Test pipeline behavior carefully after upgrade |
| R7 | Upstream repo diverges during modernization | Low | Low | Work on fork, document upstream compatibility |

## 14. Phased Implementation Plan

### Phase 1: Packaging & Dependency Modernization
- Create `pyproject.toml` with extras: `[torch]`, `[mlx]`, `[dev]`, `[train]`, `[eval]`
- Upgrade inference dependencies to current stable
- Remove CUDA-only deps from default install
- Create `docs/installation.md`

### Phase 2: Runtime Backend Abstraction
- Create `src/stablevsr/backends/` package
- Implement backend detection: mlx, torch-mps, torch-cuda, torch-cpu
- Add `--backend` CLI flag and env var
- Add `backend-info` and `doctor` commands
- Fix hardcoded CUDA references in flow_utils.py

### Phase 3: Inference Stabilization
- Restructure into `src/stablevsr/` package
- Create unified inference CLI (`stablevsr infer`)
- Wrap existing pipeline behind backend-aware service layer
- Add smoke tests

### Phase 4: MLX Apple Silicon Support
- Implement Option C: torch-mps primary, MLX scaffold
- Add honest capability reporting
- Document exactly what works and what doesn't
- Add weight conversion experiments if warranted

### Phase 5: Training/Eval Modernization (out of initial scope)
### Phase 6: Testing, CI, Quality Gates

## 15. Exact First Change Set (Phase 1)

| File | Action | Purpose |
|---|---|---|
| `pyproject.toml` | Create | Modern packaging with extras |
| `src/stablevsr/__init__.py` | Create | Package root |
| `src/stablevsr/backends/__init__.py` | Create | Backend package |
| `src/stablevsr/backends/registry.py` | Create | Backend detection & selection |
| `src/stablevsr/backends/base.py` | Create | Backend interface |
| `src/stablevsr/backends/torch_backend.py` | Create | PyTorch backend implementation |
| `src/stablevsr/backends/mlx_backend.py` | Create | MLX backend scaffold |
| `util/flow_utils.py` | Edit | Fix hardcoded `.to('cuda')` |
| `docs/installation.md` | Create | Installation docs |
| `docs/modernization_audit.md` | Create | This document |
| `requirements.txt` | Keep | Backward compatibility |
| `requirements-mac.txt` | Keep | Backward compatibility |
