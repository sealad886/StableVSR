"""MLX implementation of StableVSR pipeline components."""

from __future__ import annotations

MLX_AVAILABLE = False
try:
    import mlx.core  # noqa: F401

    MLX_AVAILABLE = True
except ModuleNotFoundError:
    pass
