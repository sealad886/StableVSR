# Developer Guide

## Prerequisites

- **Python ≥ 3.10** (3.11+ recommended)
- **Git**
- A virtual environment tool (`venv`, `virtualenv`, `conda`, etc.)

## Dev Setup

```bash
git clone https://github.com/claudiom4sir/StableVSR.git
cd StableVSR
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest`, `ruff`, `video`, and `eval` dependencies.

## Running Tests

```bash
python -m pytest tests/ -v
```

Test suite breakdown:

| File | Tests | Covers |
|---|---|---|
| `tests/test_backends.py` | 21 | Backend detection, registry, capabilities |
| `tests/test_cli.py` | 20 | CLI argument parsing, subcommands |
| `tests/test_flow_utils.py` | 16 | Optical-flow utilities |

## Linting & Formatting

```bash
ruff check src/ tests/       # lint
ruff format src/ tests/       # format
```

Config lives in `pyproject.toml` — line length 120, target Python 3.10, rules: `E`, `F`, `W`, `I`.

## Project Structure

```
StableVSR/
├── src/stablevsr/            # Installable package
│   ├── cli.py                # Entry point (stablevsr command)
│   └── backends/             # Compute backend abstraction
│       ├── base.py           # Backend ABC + BackendCapabilities
│       ├── registry.py       # Auto-detection & selection
│       ├── torch_backend.py  # PyTorch (CUDA / MPS / CPU)
│       └── mlx_backend.py    # Apple MLX
├── pipeline/                 # Diffusion pipeline (project-root, not packaged)
│   └── stablevsr_pipeline.py
├── scheduler/                # Custom DDPM scheduler (project-root)
│   └── ddpm_scheduler.py
├── util/                     # Flow utilities (project-root)
│   └── flow_utils.py
├── dataset/                  # Training dataset configs
├── tests/                    # pytest suite
├── docs/                     # Documentation
├── train.py / eval.py        # Training & evaluation scripts
└── pyproject.toml            # Build config (hatchling)
```

> **Note:** `pipeline/`, `scheduler/`, and `util/` are imported from the project root and are not part of the installed package. Only `src/stablevsr/` is packaged.

## Adding a New Backend

1. Create `src/stablevsr/backends/my_backend.py`.
2. Subclass `Backend` from `base.py` and implement all abstract methods (`name`, `is_available`, `capabilities`, `default_device`, `default_dtype_str`).
3. Register it in `registry.py` by importing your class and adding an entry to the `_BACKENDS` dict.
4. Add tests in `tests/test_backends.py`.

See [backends.md](backends.md) for architecture details.

## Package Extras

| Extra | Contents | Use case |
|---|---|---|
| `video` | `imageio`, `imageio-ffmpeg` | Video I/O |
| `train` | `basicsr`, `omegaconf`, `einops`, `bitsandbytes`\*, `xformers`\*, `wandb`, `tensorboard` | Training |
| `eval` | `torchmetrics`, `pyiqa`, `DISTS-pytorch` | Evaluation metrics |
| `dev` | `pytest`, `ruff` + `video` + `eval` | Development |
| `mlx` | `mlx` (macOS only) | MLX backend |
| `apple` | `video` + `mlx` | macOS inference |
| `torch` | *(empty — core deps cover it)* | Explicit torch marker |
| `rust-ext` | *(placeholder)* | Future Rust extensions |

\* Linux-only; excluded on macOS.

## CI

CI runs:

```bash
python -m pytest tests/ -v
ruff check src/ tests/
```

Ensure both pass before pushing.
