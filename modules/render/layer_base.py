# Standard library imports
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

# Local application imports
from modules.gl import Texture


@runtime_checkable
class HasCaptureTime(Protocol):
    """A layer that keeps the capture time of the camera frame its texture holds: a source from the
    board, or a view that forwards its source's."""
    @property
    def timestamp(self) -> float | None: ...


class LayerBase(ABC):
    """Base class for all rendering layers."""

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        pass

    @abstractmethod
    def deallocate(self) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    @property
    def texture(self) -> Texture:
        """Output texture. Override in subclasses that produce texture output."""
        raise NotImplementedError(f"{self.__class__.__name__} does not produce texture output")

    def draw(self) -> None:
        """Default implementation: draw texture to rect."""
        if self.texture.allocated:
            self.texture.draw()
