# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils.PointsAndRects import Rect
from modules.gl import Texture


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

    def draw(self, rect: Rect) -> None:
        """Default implementation: draw texture to rect."""
        if self.texture.allocated:
            self.texture.draw(rect.x, rect.y, rect.width, rect.height)