"""Backend registry: detection, selection, and enumeration."""

from __future__ import annotations

import logging
import os

from stablevsr.backends.base import Backend, BackendCapabilities
from stablevsr.backends.mlx_backend import MLXBackend
from stablevsr.backends.torch_backend import TorchBackend

log = logging.getLogger(__name__)

_BACKENDS: dict[str, type[Backend]] = {
    "torch": TorchBackend,
    "mlx": MLXBackend,
}


class BackendRegistry:
    """Manages available compute backends."""

    @staticmethod
    def get(name: str | None = None) -> Backend:
        """Get a backend by name, or auto-detect the best available.

        Resolution order:
        1. Explicit ``name`` argument
        2. ``STABLEVSR_BACKEND`` environment variable
        3. Auto-detect: mlx (if available + capable) → torch
        """
        requested = name or os.environ.get("STABLEVSR_BACKEND")

        if requested:
            key = requested.split("-")[0]
            cls = _BACKENDS.get(key)
            if cls is None:
                available = ", ".join(_BACKENDS)
                raise ValueError(
                    f"Unknown backend '{requested}'. Available: {available}"
                )

            if key == "torch":
                device = requested.split("-", 1)[1] if "-" in requested else None
                if device is not None:
                    valid = {"cpu", "cuda", "mps"}
                    if device not in valid:
                        raise ValueError(
                            f"Unknown torch device '{device}'. Valid: {', '.join(sorted(valid))}"
                        )
                backend = TorchBackend(device=device)
                log.info(
                    "Backend selected: %s (reason: explicit request '%s')",
                    backend.name(),
                    requested,
                )
                return backend
            if "-" in requested:
                raise ValueError(
                    f"Backend '{key}' does not support device suffixes; got '{requested}'"
                )
            inst = cls()
            log.info(
                "Backend selected: %s (reason: explicit request '%s')",
                inst.name(),
                requested,
            )
            return inst

        # Auto-detect: prefer MLX if it can actually do inference
        rejected: list[str] = []
        mlx = MLXBackend()
        if mlx.is_available() and mlx.capabilities().inference:
            log.info(
                "Backend selected: mlx (reason: auto-detect, MLX inference available)"
            )
            return mlx
        if mlx.is_available():
            rejected.append("mlx (available but inference not implemented)")
        else:
            rejected.append("mlx (not installed)")

        backend = TorchBackend()
        log.info(
            "Backend selected: %s (reason: auto-detect, best available torch device; "
            "rejected: %s; default dtype: %s)",
            backend.name(),
            ", ".join(rejected) if rejected else "none",
            backend.default_dtype_str(),
        )
        return backend

    @staticmethod
    def list_all() -> list[BackendCapabilities]:
        """Return capabilities for every registered backend."""
        results = []
        for cls in _BACKENDS.values():
            backend = cls()
            results.append(backend.capabilities())
        return results


def get_backend(name: str | None = None) -> Backend:
    """Convenience wrapper for ``BackendRegistry.get()``."""
    return BackendRegistry.get(name)


def list_backends() -> list[BackendCapabilities]:
    """Convenience wrapper for ``BackendRegistry.list_all()``."""
    return BackendRegistry.list_all()
