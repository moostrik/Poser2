# Standard library imports
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils.PointsAndRects import Rect
from modules.gl import Texture, Blit


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
            Blit.use(self.texture)

T = TypeVar('T')

class DataCache(Generic[T]):
    """Caches data and tracks changes."""

    def __init__(self):
        self._cached: T | None = None
        self._changed: bool = False
        self._lost: bool = False

    def update(self, new_data: T | None) -> None:
        if new_data is self._cached:
            self._changed = False
            self._lost = False
            return

        self._lost = new_data is None and self._cached is not None
        self._changed = True
        self._cached = new_data

    @property
    def data(self) -> T | None:
        return self._cached

    @property
    def changed(self) -> bool:
        return self._changed

    @property
    def idle(self) -> bool:
        return not self._changed

    @property
    def lost(self) -> bool:
        return self._lost

    @property
    def has_data(self) -> bool:
        return self._cached is not None

    @property
    def empty(self) -> bool:
        return self._cached is None