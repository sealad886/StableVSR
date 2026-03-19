"""MLX backend for Apple Silicon inference."""

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

    UNet, VAE, ControlNet, text encoder, and scheduler all run natively
    in MLX on Metal.  RAFT optical flow uses a PyTorch-CPU bridge
    (the only remaining torch runtime dependency in the MLX path).
    """

    def name(self) -> str:
        """Return the backend identifier."""
        return "mlx"

    def is_available(self) -> bool:
        """Return True if the mlx package is importable."""
        return _MLX_AVAILABLE

    def capabilities(self) -> BackendCapabilities:
        """Return MLX backend capabilities."""
        return BackendCapabilities(
            name="mlx",
            available=_MLX_AVAILABLE,
            inference=_MLX_AVAILABLE,
            training=False,
            half_precision=_MLX_AVAILABLE,
            device_name="gpu" if _MLX_AVAILABLE else "",
            notes=[
                "Native MLX inference on Apple Silicon (Metal GPU)",
                "RAFT optical flow uses a PyTorch-CPU bridge (torch required at runtime)",
                "Use 'stablevsr mlx-infer' or the MLXStableVSRPipeline API directly",
            ],
        )

    def default_device(self) -> str:
        """Return the default MLX device string, or empty if unavailable."""
        return "gpu" if _MLX_AVAILABLE else ""

    def default_dtype_str(self) -> str:
        """Return the default dtype for MLX compute."""
        return "float16"
