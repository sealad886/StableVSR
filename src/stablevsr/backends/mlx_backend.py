"""MLX backend scaffold — honest about current capabilities."""

from __future__ import annotations

from stablevsr.backends.base import Backend, BackendCapabilities

_MLX_AVAILABLE = False
try:
    import mlx.core  # noqa: F401

    _MLX_AVAILABLE = True
except ModuleNotFoundError:
    pass


class MLXBackend(Backend):
    """MLX backend for Apple Silicon.

    Currently a scaffold — the full StableVSR pipeline (custom ControlNet +
    RAFT optical flow + bidirectional sampling) cannot run on MLX yet.
    This backend reports honest capabilities and will gain features as
    MLX ecosystem support matures.
    """

    def name(self) -> str:
        """Return the backend identifier."""
        return "mlx"

    def is_available(self) -> bool:
        """Return True if the mlx package is importable."""
        return _MLX_AVAILABLE

    def capabilities(self) -> BackendCapabilities:
        """Return scaffold capabilities — inference is not yet supported."""
        return BackendCapabilities(
            name="mlx",
            available=_MLX_AVAILABLE,
            inference=False,
            training=False,
            half_precision=_MLX_AVAILABLE,
            device_name="gpu" if _MLX_AVAILABLE else "",
            notes=[
                "MLX backend is a scaffold — inference not yet implemented",
                "StableVSR requires custom ControlNet + RAFT which have no MLX equivalents",
                "Use torch-mps for Apple Silicon inference today",
            ],
        )

    def default_device(self) -> str:
        """Return the default MLX device string, or empty if unavailable."""
        return "gpu" if _MLX_AVAILABLE else ""

    def default_dtype_str(self) -> str:
        """Return the default dtype for MLX compute."""
        return "float16"
