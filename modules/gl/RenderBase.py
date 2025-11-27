from abc import ABC, abstractmethod

from OpenGL.GL import * # type: ignore

class RenderBase(ABC):
    @abstractmethod
    def allocate(self) -> None: ...
    @abstractmethod
    def deallocate(self) -> None: ...
    @abstractmethod
    def draw_main(self, width: int, height: int) -> None: ...
    @abstractmethod
    def draw_secondary(self, monitor_id: int, width: int, height: int) -> None: ...
    @abstractmethod
    def on_main_window_resize(self, width: int, height: int) -> None: ...

    def setView(self, width, height) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, width, height)