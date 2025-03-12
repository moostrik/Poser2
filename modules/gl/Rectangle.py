from OpenGL.GL import *  # type: ignore
import numpy as np

class Rectangle:
    def __init__(self, x: float = 0.0, y: float = 0.0, width: float = 0.0, height: float = 0.0, color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        self.x: float = x
        self.y: float = y
        self.width: float = width
        self.height: float = height
        self.color: tuple[float, float, float, float] = color
        self.vertex_buffer: int = 0
        self.color_buffer: int = 0
        self.allocated: bool = False

    def allocate(self) -> None:
        if not self.allocated:
            self.vertex_buffer = glGenBuffers(1)
            self.color_buffer = glGenBuffers(1)
            self._setup_vertex_buffer()
            self._setup_color_buffer()
            self.allocated = True

    def _setup_vertex_buffer(self) -> None:
        vertices: np.ndarray = np.array([
            [self.x, self.y, 0.0],
            [self.x + self.width, self.y, 0.0],
            [self.x + self.width, self.y + self.height, 0.0],
            [self.x, self.y + self.height, 0.0]
        ], dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    def _setup_color_buffer(self) -> None:
        colors: np.ndarray = np.array([
            self.color,
            self.color,
            self.color,
            self.color
        ], dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)

    def draw(self) -> None:
        if not self.allocated:
            self.allocate()

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glVertexPointer(3, GL_FLOAT, 0, None)

        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glColorPointer(4, GL_FLOAT, 0, None)

        glDrawArrays(GL_QUADS, 0, 4)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

    def set_color(self, color: tuple[float, float, float, float]) -> None:
        if self.color == color:
            return
        self.color = color
        if self.allocated:
            self._setup_color_buffer()

    def set_vertices(self, x: float, y: float, width: float, height: float) -> None:
        if self.x == x and self.y == y and self.width == width and self.height == height:
            return
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        if self.allocated:
            self._setup_vertex_buffer()

    def set_corners(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.set_vertices(x0, y0, x1 - x0, y1 - y0)

    def __del__(self) -> None:
        glDeleteBuffers(1, [self.vertex_buffer])
        glDeleteBuffers(1, [self.color_buffer])