# StableVSR — Video Super-Resolution with Temporally-Consistent Diffusion Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![ECCV 2024](https://img.shields.io/badge/ECCV-2024-orange)](https://link.springer.com/chapter/10.1007/978-3-031-73254-6_3)

[Claudio Rota](https://scholar.google.com/citations?user=HwPPoh4AAAAJ&hl=en), [Marco Buzzelli](https://scholar.google.com/citations?hl=en&user=kSFvKBoAAAAJ), [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en)

[[Paper](https://link.springer.com/chapter/10.1007/978-3-031-73254-6_3)] [[arXiv](https://arxiv.org/abs/2311.15908)] [[Poster](https://eccv.ecva.net/media/PosterPDFs/ECCV%202024/1051.png?t=1727108222.9410088)]

StableVSR enhances perceptual quality in video super-resolution using diffusion models with temporally-consistent detail synthesis. It introduces the **Temporal Conditioning Module (TCM)** with Temporal Texture Guidance and a **Frame-wise Bidirectional Sampling** strategy for high-quality, coherent frame upscaling.

<img width="640" alt="networkfull" src="https://github.com/user-attachments/assets/51390b6d-b069-49e1-a7ca-290099b2039f">

## Backend support

| Backend | Status | Platform |
|---------|--------|----------|
| `torch-cuda` | Full inference + training | Linux/Windows with NVIDIA GPU |
| `torch-mps` | Full inference | Apple Silicon Mac (arm64) |
| `torch-cpu` | Full inference (slow) | Any platform |
| `mlx` | Full inference | Apple Silicon Mac (native, optimized) |

The runtime auto-detects the best backend. Override with `--backend` or `STABLEVSR_BACKEND`.

## Quick start

### Install

```bash
git clone https://github.com/claudiom4sir/StableVSR.git
cd StableVSR
python3 -m venv .venv && source .venv/bin/activate

# Inference with video I/O
pip install -e ".[video]"

# Or: development (includes tests, linting, eval metrics)
pip install -e ".[dev]"
```

See [docs/installation.md](docs/installation.md) for platform-specific variants (`.[apple]`, `.[train]`, `.[eval]`, `.[mlx]`).

### Verify

```bash
stablevsr doctor       # check dependencies, GPU, platform
stablevsr backend-info # show available backends + capabilities
```

### Run inference

```bash
# Basic — auto-detect backend, float32
stablevsr infer --input ./lr_frames --output ./sr_output

# Apple Silicon — half precision on MPS
stablevsr infer --input ./lr_frames --output ./sr_output --fp16

# CUDA — explicit backend + bfloat16
stablevsr infer --input ./lr_frames --output ./sr_output --backend torch-cuda --dtype bfloat16

# Memory-constrained — enable tiling + CPU offload
stablevsr infer --input ./lr_frames --output ./sr_output --fp16 --vae-tiling --cpu-offload
```

Models are downloaded automatically from [HuggingFace](https://huggingface.co/claudiom4sir/StableVSR) on first run.

Input must be a directory of image sequences (PNG, JPG, BMP, TIFF, or WebP). Subdirectories are treated as separate sequences. See [docs/inference.md](docs/inference.md) for full CLI reference.

### MLX inference (Apple Silicon native)

```bash
# With preset (recommended)
stablevsr mlx-infer --input ./lr_frames --output ./sr_output --preset safe

# Long video with chunked inference
stablevsr mlx-infer --input ./lr_frames --output ./sr_output --preset safe --steps 50

# Maximum speed
stablevsr mlx-infer --input ./lr_frames --output ./sr_output --preset fast

# Custom settings
stablevsr mlx-infer --input ./lr_frames --output ./sr_output \
    --compile --ttg-start-step 10 --chunk-size 16 --chunk-overlap 4
```

Presets: `max-quality`, `safe` (recommended), `balanced`, `fast`. See [docs/quality_tradeoffs.md](docs/quality_tradeoffs.md) for details.

### Smoke test (no model download)

```bash
stablevsr infer --input ./lr_frames --output ./sr_output --smoke-test
```

Validates CLI arguments and backend selection without loading models.

## Datasets

Download the [REDS dataset](https://seungjunnah.github.io/Datasets/reds.html) (sharp + low-resolution). Data should be organized as `root/hr/sequences/frames` and `root/lr/sequences/frames`.

## Training

```bash
pip install -e ".[train]"
# Adjust dataroot in dataset/config_reds.yaml, then:
bash ./train.sh
```

Training requires ~17 GB GPU memory. See `train.sh` for configurable options.

## Evaluation

```bash
pip install -e ".[eval]"
python eval.py --gt_path YOUR_PATH_TO_GT_SEQS --out_path YOUR_OUTPUT_PATH
```

Evaluation on REDS (320×180 → 1280×720) requires ~14.5 GB.

## Apple Silicon notes

- **Primary backend**: `torch-mps` — full inference, automatic selection on Apple Silicon
- **Supported dtypes**: `float32`, `float16` (no `bfloat16` on MPS)
- **Memory tips**: Use `--vae-tiling` and `--cpu-offload` for large resolutions
- **MLX backend**: Full native inference with `stablevsr mlx-infer` — supports presets, chunked long-video processing, and `mx.compile` acceleration

See [docs/apple_silicon.md](docs/apple_silicon.md) for the full Apple Silicon guide.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/installation.md](docs/installation.md) | Platform-specific install commands |
| [docs/inference.md](docs/inference.md) | Full CLI reference, input format, dtype matrix |
| [docs/backends.md](docs/backends.md) | Backend architecture, capability matrix, extending |
| [docs/apple_silicon.md](docs/apple_silicon.md) | Apple Silicon guide, MPS limitations, MLX status |
| [docs/development.md](docs/development.md) | Dev setup, testing, project structure |
| [docs/quality_tradeoffs.md](docs/quality_tradeoffs.md) | Optimization presets, quality/speed/memory tradeoffs |
| [docs/modernization_audit.md](docs/modernization_audit.md) | Modernization audit and phased plan |

## Development

```bash
pip install -e ".[dev]"
pytest tests/          # run test suite
ruff check src/ tests/ # lint
```

## Demo videos

https://github.com/user-attachments/assets/60c5fc3b-819c-4242-bd73-e5e3b0f7beb3

https://github.com/user-attachments/assets/9fbc6fad-a088-41d9-be38-af53a8206916

https://github.com/user-attachments/assets/2f8a36f7-3b50-4eb1-baa8-e914a8931543

https://github.com/user-attachments/assets/7b379ad5-ecba-468a-811a-0a9cc4c8456d

## Citation

```bibtex
@inproceedings{rota2024enhancing,
  title={Enhancing perceptual quality in video super-resolution through temporally-consistent detail synthesis using diffusion models},
  author={Rota, Claudio and Buzzelli, Marco and van de Weijer, Joost},
  booktitle={European Conference on Computer Vision},
  pages={36--53},
  year={2024},
  organization={Springer}
}
```

## Contact

If you have any questions, please contact claudio.rota@unimib.it

