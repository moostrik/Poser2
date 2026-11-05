"""Luminance frames: the newest two frames of a flow's input as luminance, mipmapped.

Each optical flow keeps its own: it measures the motion between them, and the mip levels are the
pyramid both flows read at coarser scales.
"""
from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R8, GL_RED, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Blit, SwapFbo, Texture
from .. import FlowUtil
from .shaders import Luminance


class LuminanceFrames:
    """The newest and the previous luminance frame (R8, mipmapped)."""

    def __init__(self) -> None:
        self._frames: SwapFbo = SwapFbo()
        self._shader: Luminance = Luminance()
        self._frame_count: int = 0

    @property
    def luminance(self) -> Texture:
        """The newest luminance frame, mipmapped."""
        return self._frames

    @property
    def previous_luminance(self) -> Texture:
        """The luminance frame before it, mipmapped."""
        return self._frames.back_texture

    @property
    def ready(self) -> bool:
        """Two frames are in: motion can be measured between them."""
        return self._frames.allocated and self._frame_count >= 2

    def allocate(self, width: int, height: int) -> None:
        self._frames.allocate(width, height, GL_R8)
        for texture in (self._frames.texture, self._frames.back_texture):
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        self._shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._frames.deallocate()
        self._shader.deallocate()

    def reset(self) -> None:
        FlowUtil.zero(self._frames)
        self._frame_count = 0

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame, as luminance; the newest becomes the previous one."""
        if not self._frames.allocated or not texture.allocated:
            return

        self._frames.swap()
        self._frames.begin()
        if texture.format == GL_RED:
            Blit.use(texture)               # already one channel
        else:
            self._shader.use(texture)
        self._frames.end()

        glBindTexture(GL_TEXTURE_2D, self._frames.texture.tex_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._frame_count += 1
