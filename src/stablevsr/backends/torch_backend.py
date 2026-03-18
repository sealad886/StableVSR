"""PyTorch backend implementation (CUDA, MPS, CPU)."""

from __future__ import annotations

import torch

from stablevsr.backends.base import Backend, BackendCapabilities


class TorchBackend(Backend):
    """PyTorch backend with automatic device selection."""

    def __init__(self, device: str | None = None) -> None:
        """Initialize the backend, auto-detecting a device if none is given."""
        self._device = device or self._detect_device()

    def name(self) -> str:
        """Return the backend identifier including the active device."""
        return f"torch-{self._device}"

    def is_available(self) -> bool:
        """Return True; PyTorch is always available if this module loads."""
        return True

    def capabilities(self) -> BackendCapabilities:
        """Return capabilities based on the detected device."""
        caps = BackendCapabilities(
            name=self.name(),
            available=True,
            inference=True,
            training=(self._device != "mps"),
            half_precision=(self._device != "cpu"),
            device_name=self._device,
        )
        if self._device == "mps":
            caps.notes.append("Training not fully tested on MPS")
            caps.notes.append("Some ops may fall back to CPU")
        if self._device == "cpu":
            caps.notes.append("CPU-only: slow but always works")
        return caps

    def default_device(self) -> str:
        """Return the auto-detected or user-specified device string."""
        return self._device

    def default_dtype_str(self) -> str:
        """Return 'float32' for CPU, 'float16' otherwise."""
        if self._device == "cpu":
            return "float32"
        return "float16"

    @staticmethod
    def _detect_device() -> str:
        """Detect the best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
