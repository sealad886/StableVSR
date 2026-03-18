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
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    def default_device(self) -> str: ...

    @abstractmethod
    def default_dtype_str(self) -> str: ...
