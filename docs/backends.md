# Backend Architecture

StableVSR uses a backend abstraction layer (`src/stablevsr/backends/`) to decouple
compute-device logic from the inference and training pipeline.

## Capability Matrix

| Backend       | Available       | Inference | Training | Half Precision | Default dtype | Notes                                      |
|---------------|-----------------|-----------|----------|----------------|---------------|--------------------------------------------|
| `torch-cuda`  | if CUDA present | yes       | yes      | yes            | float16       | Full support                               |
| `torch-mps`   | if MPS present  | yes       | no       | yes            | float16       | Training not tested; some ops fall to CPU   |
| `torch-cpu`   | always          | yes       | no*      | no             | float32       | Slow but always works                      |
| `mlx`         | if mlx installed| **yes**   | no       | yes            | float16       | Primary Apple Silicon backend              |

\* CPU training is technically possible but omitted from capabilities because it is impractically slow.

## Selection Logic

Backend resolution follows a strict priority chain:

1. **Explicit argument** — `BackendRegistry.get(name)` or CLI `--backend`
2. **Environment variable** — `STABLEVSR_BACKEND`
3. **Auto-detect** — MLX (if available **and** inference-capable) → Torch (CUDA > MPS > CPU)

MLX currently reports `inference=False`, so auto-detect always falls through to Torch.

```
┌──────────────────────┐
│ --backend provided?  │──yes──▸ Use that backend
└──────────┬───────────┘
           no
┌──────────────────────────────┐
│ STABLEVSR_BACKEND env set?  │──yes──▸ Use that backend
└──────────┬───────────────────┘
           no
┌──────────────────────────────────────┐
│ MLX installed + inference capable?  │──yes──▸ Use MLX
└──────────┬───────────────────────────┘
           no
┌─────────────────────────┐
│ Torch: CUDA > MPS > CPU │
└─────────────────────────┘
```

## Querying Backends

```bash
stablevsr backend-info
```

Prints every registered backend with its availability, device, capability flags, and notes.
Example output on Apple Silicon:

```
[torch-mps] AVAILABLE
  Device:         mps
  Inference:      yes
  Training:       no
  Half precision: yes
  Note: Training not fully tested on MPS
  Note: Some ops may fall back to CPU

[mlx] AVAILABLE
  Device:         gpu
  Inference:      no
  Training:       no
  Half precision: yes
  Note: MLX backend is a scaffold — inference not yet implemented
  Note: StableVSR requires custom ControlNet + RAFT which have no MLX equivalents
  Note: Use torch-mps for Apple Silicon inference today
```

## Forcing a Backend

### CLI flag

```bash
stablevsr infer --backend torch-cpu --input ./lr --output ./sr
stablevsr infer --backend torch-mps --input ./lr --output ./sr
```

The `--backend` value is `<backend>` or `<backend>-<device>` for torch
(`torch-cuda`, `torch-mps`, `torch-cpu`). Invalid device names raise immediately.

### Environment variable

```bash
export STABLEVSR_BACKEND=torch-cuda
stablevsr infer --input ./lr --output ./sr
```

The env var uses the same format. CLI `--backend` takes precedence over the env var.

### Programmatic

```python
from stablevsr.backends import get_backend

backend = get_backend("torch-mps")
print(backend.default_device())   # "mps"
print(backend.default_dtype_str())  # "float16"
```

## Adding a New Backend

1. Create `src/stablevsr/backends/my_backend.py`.
2. Subclass `Backend` and implement the five abstract methods:

   ```python
   from stablevsr.backends.base import Backend, BackendCapabilities

   class MyBackend(Backend):
       def name(self) -> str: ...
       def is_available(self) -> bool: ...
       def capabilities(self) -> BackendCapabilities: ...
       def default_device(self) -> str: ...
       def default_dtype_str(self) -> str: ...
   ```

3. Register it in `registry.py`:

   ```python
   from stablevsr.backends.my_backend import MyBackend
   _BACKENDS["my"] = MyBackend
   ```

4. Auto-detect logic lives in `BackendRegistry.get()`. Update the fallback chain
   there if the new backend should participate in auto-detection.

## MLX Status

**Current state: full inference pipeline operational on Apple Silicon.**

The MLX backend (`src/stablevsr/mlx/`) provides:

- **UNet, VAE, ControlNet** — all ported to native MLX with Metal GPU acceleration
- **DDPM scheduler** — MLX-native reimplementation
- **Tiled VAE decode** — auto-detected for large latents (>4096 px area)
- **`mx.compile()`** — JIT shader fusion for 2.15× UNet speedup
- **Temporal guidance step-skip** — configurable `ttg_start_step` to trade coherence for speed

RAFT optical flow uses a PyTorch CPU bridge (tensor round-trip). This is the
only remaining torch dependency in the MLX inference path.

See [benchmark_results.md](benchmark_results.md) for performance data and
[apple_silicon.md](apple_silicon.md) for usage guide.

## Runtime Logging

Backend selection is logged via Python's `logging` module at `INFO` level.
Enable verbose output to see it:

```bash
stablevsr infer --input ./lr --output ./sr -v
```

Log messages include:

- **Selected backend and reason** — e.g. `Backend selected: torch-mps (reason: auto-detect, best available torch device; rejected: mlx (available but inference not implemented); default dtype: float16)`
- **Rejected alternatives** — backends that were considered but skipped, with the reason.
- **Effective dtype** — the resolved dtype after applying device restrictions (e.g. `Effective dtype: float16 (device: mps)`).

When a dtype is unsupported on the active device, a warning is printed and the dtype
falls back to `float32`:

```
[WARN] dtype 'bfloat16' not supported on mps; falling back to float32 (allowed: float16, float32)
```
