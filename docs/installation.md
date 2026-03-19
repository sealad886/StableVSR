# Installation

## Quick Start (Apple Silicon / macOS)

```bash
# Clone the repository
git clone https://github.com/claudiom4sir/StableVSR.git
cd StableVSR

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install for inference with video I/O
pip install -e ".[video]"

# Verify installation
stablevsr doctor
stablevsr backend-info
```

## Install Variants

### Inference only (minimal)
```bash
pip install -e .
```

### Inference with video file support
```bash
pip install -e ".[video]"
```

### Development (includes tests and linting)
```bash
pip install -e ".[video,dev]"
```

### Training (CUDA required)
```bash
pip install -e ".[train]"
```

### Evaluation metrics
```bash
pip install -e ".[eval]"
```

### MLX (Apple Silicon native inference)
```bash
pip install -e ".[mlx]"
```

Requires macOS with Apple Silicon. RAFT optical flow still requires PyTorch at runtime.

## Backend Selection

StableVSR auto-detects the best available backend:

| Platform | Default Backend | Notes |
|---|---|---|
| Apple Silicon Mac | `torch-mps` | Full inference support; or use `mlx-infer` for native MLX |
| NVIDIA GPU | `torch-cuda` | Full inference + training |
| CPU only | `torch-cpu` | Slow but always works |

Override with environment variable:
```bash
export STABLEVSR_BACKEND=torch-mps
```

Or via CLI flag (when inference subcommand is added):
```bash
stablevsr infer --backend torch-cpu ...
```

## Verifying Your Setup

```bash
# Check all dependencies and backends
stablevsr doctor

# See available backends and their capabilities
stablevsr backend-info
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (with MPS support for Apple Silicon)
- ~6GB disk space for model weights
- ~8GB RAM for inference (more for higher resolutions)

## Pretrained Models

Models are available on [HuggingFace](https://huggingface.co/claudiom4sir/StableVSR)
and are downloaded automatically via `from_pretrained('claudiom4sir/StableVSR')`.

For offline use, download and point to the local directory:
```bash
huggingface-cli download claudiom4sir/StableVSR --local-dir models/StableVSR
```
