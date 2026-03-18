"""Abstract base interface for compute backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BackendCapabilities:
    """Describes what a backend can do."""

    name: str
    available: bool
    inference: bool = False
    training: bool = False
    half_precision: bool = False
    device_name: str = ""
    notes: list[str] = field(default_factory=list)


class Backend(ABC):
    """Base class for compute backends."""

    @abstractmethod
    def name(self) -> str:
        """Return the backend identifier string."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend's dependencies are installed and usable."""
        ...

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return a capabilities descriptor for this backend."""
        ...

    @abstractmethod
    def default_device(self) -> str:
        """Return the default device string for this backend."""
        ...

    @abstractmethod
    def default_dtype_str(self) -> str:
        """Return the default dtype name (e.g. 'float16') for this backend."""
        ...
