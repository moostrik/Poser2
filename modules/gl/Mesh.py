from OpenGL.GL import *  # type: ignore
import numpy as np
from threading import Lock

class Mesh:
    def __init__(self) -> None:
        self.vertex_buffer: int =           0
        self.index_buffer: int =            0
        self.color_buffer: int =            0

        self.vertices: np.ndarray | None =  None
        self.indices: np.ndarray | None =   None
        self.colors: np.ndarray | None =    None

        self.update_vertices: bool =        False
        self.update_indices: bool =         False
        self.update_colors: bool =          False

        self.allocated =                    False

        self._mutex: Lock = Lock()

    def allocate(self) -> None:
        self.vertex_buffer =                glGenBuffers(1)
        self.index_buffer =                 glGenBuffers(1)
        self.color_buffer =                 glGenBuffers(1)

        self.allocated =                    True

        self.update()

    def deallocate(self) -> None :
        # delete the buffers?

        if not self.allocated: return

        if self.vertex_buffer:
            glDeleteBuffers(1, [self.vertex_buffer])
        if self.index_buffer:
            glDeleteBuffers(1, [self.index_buffer])
        if self.color_buffer:
            glDeleteBuffers(1, [self.color_buffer])


    def bind(self) -> None:
        if self.vertices is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
            glVertexPointer(3, GL_FLOAT, 0, None)

        if self.colors is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
            glColorPointer(3, GL_FLOAT, 0, None)

        if self.indices is not None:
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)

    def unbind(self) -> None:
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, x, y, w, h) -> None:
        if not self.allocated:
            return
        if self.vertices is None or self.indices is None:
            print('Mesh not initialized')
            return

        self.bind()

        glEnableClientState(GL_VERTEX_ARRAY)
        if self.colors is not None:
            glEnableClientState(GL_COLOR_ARRAY)

        glPushMatrix()
        glTranslatef(x, y, 0.0)
        glScalef(w, h, 1.0)

        glLineWidth(3.0)

        glDrawElements(GL_LINES, len(self.indices), GL_UNSIGNED_INT, None)

        glPopMatrix()

        glDisableClientState(GL_VERTEX_ARRAY)
        if self.colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)

        self.unbind()

    def update(self) -> None:
        if not self.allocated:
            self.allocate()

        with self._mutex:

            if self.update_vertices and self.vertices is not None:
                glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
                glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                self.update_vertices = False

            if self.update_indices and self.indices is not None:
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
                self.update_indices = False

            if self.update_colors and self.colors is not None:
                glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
                glBufferData(GL_ARRAY_BUFFER, self.colors.nbytes, self.colors, GL_STATIC_DRAW)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                self.update_colors = False


    def set_vertices(self, vertices: np.ndarray) -> None:
        if vertices.shape[1] == 2:
            vertices = np.concatenate((vertices, np.zeros((vertices.shape[0], 1), dtype=np.float32)), axis=1)
        if self.vertices is not None and np.array_equal(self.vertices, vertices):
            return
        with self._mutex:
            self.update_vertices = True
            self.vertices = vertices

    def set_indices(self, indices: np.ndarray) -> None:
        if self.indices is not None and np.array_equal(self.indices, indices):
            return
        with self._mutex:
            self.update_indices = True
            self.indices = indices.astype(np.uint32)

    def set_colors(self, colors: np.ndarray) -> None:
        if colors.ndim == 1:
            colors = np.repeat(colors[:, np.newaxis], 3, axis=1)
        if self.colors is not None and np.array_equal(self.colors, colors):
            return
        with self._mutex:
            self.update_colors = True
            self.colors = colors

    def isInitialized(self) -> bool:
        return self.allocated and self.vertices is not None and self.indices is not None

    def __del__(self) -> None:
        self.deallocate()