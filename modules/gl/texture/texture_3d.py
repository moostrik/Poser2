"""Texture3D: a 3D texture (GL_TEXTURE_3D), filtered trilinearly, and its ping-pong pair."""
from OpenGL.GL import Constant, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, GL_TEXTURE_WRAP_R, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, glBindTexture, glTexImage3D, glTextureParameterfv, glTextureParameteri, glTextureSubImage3D
import numpy as np

from .swap import TextureSwap
from .texture_base import TextureBase


class Texture3D(TextureBase):
    """A 3D texture (GL_TEXTURE_3D) of width × height × depth texels, its wrap on all three axes."""

    TARGET = GL_TEXTURE_3D
    WRAP_AXES = (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R)

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """`interpolation`: GL_LINEAR for trilinear, or GL_NEAREST; `wrap`: on all three axes; `border_color`: RGBA
        with GL_CLAMP_TO_BORDER."""
        super().__init__(interpolation, wrap, border_color)
        self.depth: int = 0

    def allocate(self, width: int, height: int, depth: int, internal_format: Constant | int) -> None:
        """The texture made at `width` × `height` × `depth` texels in `internal_format`, cleared to 0."""
        if not self._create(width, height, internal_format):
            return
        self.depth = depth
        glTexImage3D(GL_TEXTURE_3D, 0, self.internal_format, self.width, self.height, self.depth, 0,
                     self.format, self.data_type, None)
        glBindTexture(GL_TEXTURE_3D, 0)
        self.allocated = True
        self.clear()

    def upload(self, array: np.ndarray) -> None:
        """`array`, (depth, height, width[, channels]) of the volume's size in its upload type, written into it
        (glTextureSubImage3D)."""
        texels = self._upload_contiguous(array)
        if texels is not None:
            glTextureSubImage3D(self.tex_id, 0, 0, 0, 0, self.width, self.height, self.depth, self.format,
                                self.data_type, texels)

    def set_wrap_z(self, wrap: int, border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Change Z (R) wrap mode at runtime independently from XY (GL thread only)."""
        if not self.allocated:
            return
        glTextureParameteri(self.tex_id, GL_TEXTURE_WRAP_R, wrap)
        glTextureParameterfv(self.tex_id, GL_TEXTURE_BORDER_COLOR, border_color)

    def deallocate(self) -> None:
        """Release texture resources."""
        super().deallocate()
        self.depth = 0

    def read_to_numpy(self, flip: bool = False) -> np.ndarray | None:
        """The volume read back to the CPU, (depth, height, width[, channels]); `flip` turns each layer's rows."""
        pixels = self._read((self.depth, self.height, self.width))
        if pixels is not None and flip:
            pixels = pixels[:, ::-1, :]
        return pixels


class SwapTexture3D(TextureSwap[Texture3D], Texture3D):
    """Double-buffered 3D texture for ping-pong compute operations: no FBO, every read and write through
    imageLoad/imageStore in compute shaders."""

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        TextureSwap.__init__(self, Texture3D(interpolation, wrap, border_color), Texture3D(interpolation, wrap, border_color))

    @property
    def depth(self) -> int:
        return self.current.depth

    def allocate(self, width: int, height: int, depth: int, internal_format: Constant | int) -> None:
        """Both buffers made."""
        for buffer in self._buffers:
            buffer.allocate(width, height, depth, internal_format)

    def set_wrap_z(self, wrap: int, border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Change Z wrap mode on both buffers at runtime (GL thread only)."""
        for buffer in self._buffers:
            buffer.set_wrap_z(wrap, border_color)
