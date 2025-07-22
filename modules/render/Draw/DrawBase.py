# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils.PointsAndRects import Rect

class DrawBase(ABC):
    @abstractmethod
    def allocate(self, width: int, height: int, internal_format: int) -> None: ...
    @abstractmethod
    def deallocate(self) -> None: ...
    @abstractmethod
    def update(self, only_if_dirty: bool) -> None: ...
    @abstractmethod
    def draw(self, rect:Rect) -> None: ...

    @staticmethod
    def setView(width, height) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, width, height)
