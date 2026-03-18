"""Runtime backend detection and selection for StableVSR."""

from stablevsr.backends.registry import BackendRegistry, get_backend, list_backends

__all__ = ["BackendRegistry", "get_backend", "list_backends"]
