"""Texture2DArray - 2D texture array (GL_TEXTURE_2D_ARRAY) for volumetric data.

Each layer is a standard 2D texture with optimal 2D tiling/swizzle layout,
avoiding the 3D tile pathology that causes performance cliffs at certain
GL_TEXTURE_3D volume sizes.

Unlike GL_TEXTURE_3D, sampler2DArray does NOT trilinearly interpolate
between layers — callers must handle Z interpolation manually in GLSL.

Usage:
    density_tex = Texture2DArray(wrap=GL_CLAMP_TO_BORDER, border_color=(0,0,0,0))
    density_tex.allocate(3840, 2160, 28, GL_RGBA16F)
"""
from OpenGL.GL import Constant, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, glBindTexture, glTexImage3D
import numpy as np

from .swap import Swap
from .texture_base import TextureBase


class Texture2DArray(TextureBase):
    """2D array OpenGL texture (GL_TEXTURE_2D_ARRAY) for volumetric data.

    Same interface as Texture3D but backed by GL_TEXTURE_2D_ARRAY for
    optimal per-layer 2D memory tiling.

    Wrap modes control boundary conditions (S/T only — R axis is layer index):
        GL_CLAMP_TO_BORDER + (0,0,0,0) → Dirichlet / no-slip (density)
        GL_CLAMP_TO_EDGE              → Neumann / zero-gradient
    """

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


class SwapTexture2DArray(Swap[Texture2DArray], Texture2DArray):
    """Double-buffered 2D array texture for ping-pong compute operations."""

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        Swap.__init__(self, Texture2DArray(interpolation, wrap, border_color),
                      Texture2DArray(interpolation, wrap, border_color))

    @property
    def depth(self) -> int:
        return self.current.depth

    def allocate(self, width: int, height: int, depth: int, internal_format: Constant | int) -> None:
        """Both buffers made."""
        for buffer in self._buffers:
            buffer.allocate(width, height, depth, internal_format)
