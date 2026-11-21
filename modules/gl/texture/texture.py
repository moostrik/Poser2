from OpenGL.GL import Constant, GL_CLAMP_TO_EDGE, GL_LINEAR, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, glBindTexture, glTexImage2D, glTextureSubImage2D
import numpy as np

from modules.utils import Quad, Rect
from ..state.context import context
from .texture_base import TextureBase


class Texture(TextureBase):
    """A 2D texture (GL_TEXTURE_2D)."""

    TARGET = GL_TEXTURE_2D
    WRAP_AXES = (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T)

    def __init__(self,
                 interpolation: int = GL_LINEAR,
                 wrap: int = GL_CLAMP_TO_EDGE,
                 border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
                 bgr: bool = False) -> None:
        """`interpolation`: GL_LINEAR or GL_NEAREST; `wrap`: on both axes (GL_CLAMP_TO_EDGE, GL_REPEAT,
        GL_MIRRORED_REPEAT, GL_CLAMP_TO_BORDER); `border_color`: RGBA with GL_CLAMP_TO_BORDER; `bgr`: its uploads in
        BGR order."""
        super().__init__(interpolation, wrap, border_color, bgr)

    def allocate(self, width: int, height: int, internal_format: Constant | int) -> None:
        """The texture made at `width` × `height` pixels in `internal_format` (GL_RGB8, GL_RGBA16F, …)."""
        if not self._create(width, height, internal_format):
            return
        glTexImage2D(GL_TEXTURE_2D, 0, self.internal_format, self.width, self.height, 0, self.format, self.data_type, None)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.allocated = True

    def upload(self, array: np.ndarray) -> None:
        """`array`, (height, width[, channels]) of the texture's size in its upload type, written into it as it is,
        row 0 at GL's first row (glTextureSubImage2D)."""
        pixels = self._upload_contiguous(array)
        if pixels is not None:
            glTextureSubImage2D(self.tex_id, 0, 0, 0, self.width, self.height, self.format, self.data_type, pixels)

    def read_to_numpy(self, flip: bool = True, level: int = 0) -> np.ndarray | None:
        """The texture's mip `level` read back to the CPU, (height, width[, channels]) of that level, halved per
        level; `flip` turns GL's rows to the image's."""
        pixels = self._read((max(1, self.height >> level), max(1, self.width >> level)), level)
        if pixels is not None and flip:
            pixels = np.flipud(pixels)
        return pixels

    def draw(self, dst: Rect | None = None, src: Rect | Quad | None = None, opacity: float = 1.0) -> None:
        """This texture's `src` drawn at `dst` in the bound target, both 0..1 with the origin top-left (all of it
        over all of the target without them); a `src` `Rect` with a negative width or height draws it mirrored on
        that axis, a `src` `Quad` of four corners draws that part turned. Through the current context's `Blit`, with
        the blend state as set."""
        context().blit.use(self, dst, src, opacity)
