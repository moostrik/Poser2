"""Flow utility functions for FBO operations.

Ported from ofxFlowTools ftUtil.h/cpp
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Texture import Texture
from modules.gl.Shader import draw_quad


class FlowUtil:
    """Static utility methods for flow FBO operations."""

    @staticmethod
    def zero(fbo: Fbo | SwapFbo) -> None:
        """Clear FBO to black/zero.

        Args:
            fbo: Fbo or SwapFbo to clear
        """
        if isinstance(fbo, SwapFbo):
            fbo.clear_all(0.0, 0.0, 0.0, 0.0)
        else:
            fbo.clear(0.0, 0.0, 0.0, 0.0)

    @staticmethod
    def one(fbo: Fbo | SwapFbo) -> None:
        """Fill FBO with white/one.

        Args:
            fbo: Fbo or SwapFbo to fill
        """
        if isinstance(fbo, SwapFbo):
            fbo.clear_all(1.0, 1.0, 1.0, 1.0)
        else:
            fbo.clear(1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def stretch(dst_fbo: Fbo, src_texture: Texture) -> None:
        """Copy/stretch source texture to destination FBO.

        Args:
            dst_fbo: Destination FBO
            src_texture: Source texture
        """
        if not dst_fbo.allocated:
            return

        glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo.fbo_id)
        glViewport(0, 0, dst_fbo.width, dst_fbo.height)

        glDisable(GL_BLEND)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src_texture.tex_id)

        # Simple textured quad using fixed function
        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1.0, 1.0)
        glEnd()

        glDisable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glEnable(GL_BLEND)

    @staticmethod
    def copy(dst_fbo: Fbo, src_fbo: Fbo) -> None:
        """Copy source FBO to destination FBO.

        Args:
            dst_fbo: Destination FBO
            src_fbo: Source FBO
        """
        FlowUtil.stretch(dst_fbo, src_fbo)

    @staticmethod
    def add(dst_fbo: SwapFbo, src_texture: Texture, strength: float = 1.0) -> None:
        """Add source texture to destination SwapFbo with strength multiplier.

        Uses ping-pong buffer: reads from current state, writes to swapped state.

        Args:
            dst_fbo: Destination SwapFbo (will be swapped)
            src_texture: Source texture to add
            strength: Multiplier for source texture (default 1.0)
        """
        # Lazy init shader
        if not hasattr(FlowUtil, '_add_shader'):
            from .shaders.AddMultiplied import AddMultiplied
            FlowUtil._add_shader = AddMultiplied()
            FlowUtil._add_shader.allocate()

        # Swap and add: curr = prev + (src * strength)
        dst_fbo.swap()
        FlowUtil._add_shader.use(
            dst_fbo, dst_fbo.back_texture, src_texture,
            strength0=1.0, strength1=strength
        )

    @staticmethod
    def get_num_channels(internal_format: int) -> int:
        """Get number of channels from OpenGL internal format.

        Args:
            internal_format: OpenGL internal format constant

        Returns:
            Number of channels (1-4)
        """
        format_channels = {
            GL_R8: 1, GL_R16: 1, GL_R16F: 1, GL_R32F: 1,
            GL_RG8: 2, GL_RG16: 2, GL_RG16F: 2, GL_RG32F: 2,
            GL_RGB8: 3, GL_RGB16: 3, GL_RGB16F: 3, GL_RGB32F: 3,
            GL_RGBA8: 4, GL_RGBA16: 4, GL_RGBA16F: 4, GL_RGBA32F: 4,
        }
        return format_channels.get(internal_format, 4)
