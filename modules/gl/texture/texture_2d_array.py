"""Texture2DArray: a 2D array texture (GL_TEXTURE_2D_ARRAY), layers of 2D textures, and its ping-pong pair."""
from OpenGL.GL import Constant, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, glBindTexture, glTexImage3D
import numpy as np

from .swap import TextureSwap
from .texture_base import TextureBase


class Texture2DArray(TextureBase):
    """A 2D array texture (GL_TEXTURE_2D_ARRAY) of width × height pixels × depth layers, as `Texture3D`'s interface;
    filtered within a layer, not between layers, its wrap on x and y."""

    TARGET = GL_TEXTURE_2D_ARRAY
    WRAP_AXES = (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T)     # the layer index is not spatial

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """`interpolation`: GL_LINEAR for bilinear within a layer, or GL_NEAREST; `wrap`: on the S and T axes;
        `border_color`: RGBA with GL_CLAMP_TO_BORDER."""
        super().__init__(interpolation, wrap, border_color)
        self.depth: int = 0

    def allocate(self, width: int, height: int, depth: int, internal_format: Constant | int) -> None:
        """The texture made at `width` × `height` pixels × `depth` layers in `internal_format`, cleared to 0."""
        if not self._create(width, height, internal_format):
            return
        self.depth = depth
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, self.internal_format, self.width, self.height, self.depth, 0,
                     self.format, self.data_type, None)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        self.allocated = True
        self.clear()

    def deallocate(self) -> None:
        """Release texture resources."""
        super().deallocate()
        self.depth = 0

    def read_to_numpy(self, flip: bool = False) -> np.ndarray | None:
        """The layers read back to the CPU, (depth, height, width[, channels]); `flip` turns each layer's rows."""
        pixels = self._read((self.depth, self.height, self.width))
        if pixels is not None and flip:
            pixels = pixels[:, ::-1, :]
        return pixels


class SwapTexture2DArray(TextureSwap[Texture2DArray], Texture2DArray):
    """Double-buffered 2D array texture for ping-pong compute operations."""

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        TextureSwap.__init__(self, Texture2DArray(interpolation, wrap, border_color),
                             Texture2DArray(interpolation, wrap, border_color))

    @property
    def depth(self) -> int:
        return self.current.depth

    def allocate(self, width: int, height: int, depth: int, internal_format: Constant | int) -> None:
        """Both buffers made."""
        for buffer in self._buffers:
            buffer.allocate(width, height, depth, internal_format)
