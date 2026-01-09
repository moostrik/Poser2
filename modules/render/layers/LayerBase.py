# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils.PointsAndRects import Rect
from modules.gl import Texture


class LayerBase(ABC):

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        pass

    @abstractmethod
    def deallocate(self) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def draw(self, rect: Rect) -> None: ...

    # @property
    # @abstractmethod
    # def texture(self) -> Texture:...

