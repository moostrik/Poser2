"""Texture3D - 3D texture (GL_TEXTURE_3D) for volumetric fluid simulation.

Supports trilinear filtering between depth layers and per-field wrap modes
for implicit boundary conditions (no explicit obstacle border needed).

Usage:
    velocity_tex = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0,0,0,0))
    velocity_tex.allocate(256, 192, 32, GL_RGBA16F)  # u,v,w + spare

    pressure_tex = Texture3D(wrap=GL_CLAMP_TO_EDGE)   # Neumann BC
    pressure_tex.allocate(256, 192, 32, GL_R16F)
"""
from OpenGL.GL import Constant, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, GL_TEXTURE_WRAP_R, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, glBindTexture, glTexImage3D, glTexParameterfv, glTexParameteri
import numpy as np

from .swap import Swap
from .texture_base import TextureBase


class Texture3D(TextureBase):
    """3D OpenGL texture (GL_TEXTURE_3D) for volumetric data.

    Wrap modes control boundary conditions:
        GL_CLAMP_TO_BORDER + (0,0,0,0) → Dirichlet / no-slip (velocity, density)
        GL_CLAMP_TO_EDGE              → Neumann / zero-gradient (pressure, temperature)
        GL_CLAMP_TO_BORDER + (1,0,0,0) → out-of-bounds = obstacle (obstacle mask)
    """

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

    def set_wrap_z(self, wrap: int, border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """Change Z (R) wrap mode at runtime independently from XY (GL thread only)."""
        if not self.allocated:
            return
        glBindTexture(GL_TEXTURE_3D, self.tex_id)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, wrap)
        glTexParameterfv(GL_TEXTURE_3D, GL_TEXTURE_BORDER_COLOR, border_color)
        glBindTexture(GL_TEXTURE_3D, 0)

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


class SwapTexture3D(Swap[Texture3D], Texture3D):
    """Double-buffered 3D texture for ping-pong compute operations: no FBO, every read and write through
    imageLoad/imageStore in compute shaders."""

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        Swap.__init__(self, Texture3D(interpolation, wrap, border_color), Texture3D(interpolation, wrap, border_color))

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
