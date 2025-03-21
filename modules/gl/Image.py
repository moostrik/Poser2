from OpenGL.GL import * # type: ignore
from modules.gl.Texture import Texture, draw_quad
import numpy as np
from threading import Lock

class Image(Texture):
    def __init__(self) -> None:
        super().__init__()
        self._image: np.ndarray | None = None
        self._needs_update: bool = False
        self._mutex: Lock = Lock()

    def set_image(self, image: np.ndarray) -> None:
        with self._mutex:
            if self._image is None or self._image is not image:
                self._image = image
                self._needs_update = True

    def update(self) -> None:
        with self._mutex:
            if self._needs_update and self._image is not None:
                self.set_from_image(self._image)
                self._needs_update = False

    def draw(self, x, y, w, h) -> None : #override
        self.bind()
        draw_quad(x, y, w, h, True)
        self.unbind()
