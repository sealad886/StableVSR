"""Tests for backend detection and selection."""

import pytest

from stablevsr.backends import get_backend, list_backends
from stablevsr.backends.base import Backend, BackendCapabilities
from stablevsr.backends.mlx_backend import MLXBackend
from stablevsr.backends.torch_backend import TorchBackend


class TestTorchBackend:
    def test_is_available(self):
        backend = TorchBackend()
        assert backend.is_available()

    def test_name_includes_device(self):
        backend = TorchBackend(device="cpu")
        assert backend.name() == "torch-cpu"

    def test_cpu_defaults_to_float32(self):
        backend = TorchBackend(device="cpu")
        assert backend.default_dtype_str() == "float32"

    def test_capabilities_returns_correct_type(self):
        backend = TorchBackend()
        caps = backend.capabilities()
        assert isinstance(caps, BackendCapabilities)
        assert caps.available is True
        assert caps.inference is True

    def test_auto_detect_returns_valid_device(self):
        backend = TorchBackend()
        assert backend.default_device() in ("cuda", "mps", "cpu")


class TestMLXBackend:
    def test_name(self):
        backend = MLXBackend()
        assert backend.name() == "mlx"

    def test_capabilities_reports_inference(self):
        backend = MLXBackend()
        caps = backend.capabilities()
        assert caps.inference is True
        assert caps.training is False

    def test_notes_mention_raft_bridge(self):
        backend = MLXBackend()
        caps = backend.capabilities()
        assert any("RAFT" in note for note in caps.notes)


class TestRegistry:
    def test_get_torch_explicit(self):
        backend = get_backend("torch")
        assert isinstance(backend, TorchBackend)

    def test_get_torch_cpu_explicit(self):
        backend = get_backend("torch-cpu")
        assert isinstance(backend, TorchBackend)
        assert backend.default_device() == "cpu"

    def test_get_mlx_explicit(self):
        backend = get_backend("mlx")
        assert isinstance(backend, MLXBackend)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_auto_detect_returns_backend(self):
        backend = get_backend()
        assert isinstance(backend, Backend)

    def test_list_backends_returns_all(self):
        results = list_backends()
        assert len(results) >= 2
        names = {r.name for r in results}
        assert "mlx" in names


class TestRegistryDeviceSuffixValidation:
    """Adversarial tests for device suffix validation in registry."""

    def test_torch_bogus_device_raises(self):
        with pytest.raises(ValueError, match="Unknown torch device"):
            get_backend("torch-bogus")

    def test_torch_gpu_raises(self):
        """GPU is not valid—must use 'cuda' or 'mps'."""
        with pytest.raises(ValueError, match="Unknown torch device"):
            get_backend("torch-gpu")

    def test_mlx_with_suffix_raises(self):
        with pytest.raises(ValueError, match="does not support device suffixes"):
            get_backend("mlx-gpu")

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("STABLEVSR_BACKEND", "torch-cpu")
        backend = get_backend()
        assert isinstance(backend, TorchBackend)
        assert backend.default_device() == "cpu"

    def test_env_var_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("STABLEVSR_BACKEND", "nonexistent")
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend()


class TestMLXDefaultDevice:
    """Adversarial test for MLX default_device when unavailable."""

    def test_default_device_empty_when_unavailable(self, monkeypatch):
        """If MLX is not available, default_device should return empty string."""
        import stablevsr.backends.mlx_backend as mod

        monkeypatch.setattr(mod, "_MLX_AVAILABLE", False)
        backend = MLXBackend()
        assert backend.default_device() == ""

    def test_default_device_gpu_when_available(self, monkeypatch):
        import stablevsr.backends.mlx_backend as mod

        monkeypatch.setattr(mod, "_MLX_AVAILABLE", True)
        backend = MLXBackend()
        assert backend.default_device() == "gpu"
