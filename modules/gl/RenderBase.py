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
        """Set orthographic projection with top-left origin.
        
        COORDINATE SYSTEM CONVENTION:
        - Origin (0, 0) is at TOP-LEFT of screen
        - Y increases DOWNWARD (screen/UI convention)
        - This matches NumPy/image arrays and window coordinates
        - NOT standard OpenGL (which has bottom-left origin)
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, width, height, 0, -1, 1)  # Top-left origin: Y goes top→bottom

        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, width, height)