# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils.PointsAndRects import Rect

class BaseRender(ABC):
    _key: str | None = None

    @classmethod
    def key(cls) -> str:
        """Auto-generate render type identifier from class name if not explicitly set"""
        if cls._key is not None:
            return cls._key

        # Default: strip 'Render' from class name and uppercase
        cls._key = cls.__name__
        if cls._key.endswith("Render"):
            cls._key = cls._key[:-6]  # Remove "Render"
            cls._key = cls._key.upper()
        return cls._key


    @abstractmethod
    def allocate(self, width: int, height: int, internal_format: int) -> None: ...
    @abstractmethod
    def deallocate(self) -> None: ...
    @abstractmethod
    def update(self) -> None: ...
    @abstractmethod
    def draw(self, rect:Rect) -> None: ...

    @staticmethod
    def setView(width, height) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, width, height)
