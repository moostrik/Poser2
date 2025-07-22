# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils.PointsAndRects import Rect

class BaseRender(ABC):
    key: str | None = None

    @classmethod
    def get_key(cls) -> str:
        """Auto-generate render type identifier from class name if not explicitly set"""
        if cls.key is not None:
            return cls.key

        # Default: strip 'Render' from class name and uppercase
        cls.key = cls.__name__
        if cls.key.endswith("Render"):
            cls.key = cls.key[:-6]  # Remove "Render"
            cls.key = cls.key.upper()
        return cls.key


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
