# Standard library imports
from abc import ABC, abstractmethod

# Local application imports
from modules.utils.PointsAndRects import Rect

class RendererBase(ABC):
    @abstractmethod
    def allocate(self) -> None: ...
    @abstractmethod
    def deallocate(self) -> None: ...
    @abstractmethod
    def update(self) -> None: ...
    @abstractmethod
    def draw(self, rect:Rect) -> None: ...
