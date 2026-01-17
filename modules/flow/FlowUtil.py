"""Flow utility functions for FBO operations.

Ported from ofxFlowTools ftUtil.h/cpp
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Fbo import Fbo, SwapFbo
from modules.gl.Texture import Texture


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
    def blit(dst_fbo: Fbo, src_texture: Texture) -> None:
        """Copy/stretch source texture to destination FBO.

        Args:
            dst_fbo: Destination FBO
            src_texture: Source texture
        """
        if not dst_fbo.allocated:
            return

        # Lazy init shader
        if not hasattr(FlowUtil, '_stretch_shader'):
            from .shaders.Blit import Blit
            FlowUtil._stretch_shader = Blit()
            FlowUtil._stretch_shader.allocate()

        dst_fbo.begin()
        FlowUtil._stretch_shader.use(src_texture)
        dst_fbo.end()

    @staticmethod
    def copy(dst_fbo: Fbo, src_fbo: Fbo) -> None:
        """Copy source FBO to destination FBO.

        Args:
            dst_fbo: Destination FBO
            src_fbo: Source FBO
        """
        FlowUtil.blit(dst_fbo, src_fbo)

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
            from .shaders.Blend import Blend
            FlowUtil._add_shader = Blend()
            FlowUtil._add_shader.allocate()

        # Swap and add: curr = prev + (src * strength)
        dst_fbo.swap()
        dst_fbo.begin()
        FlowUtil._add_shader.use(dst_fbo.back_texture, src_texture, 1.0, strength)
        dst_fbo.end()

    @staticmethod
    def set(dst_fbo: SwapFbo, src: Texture, strength: float = 1.0) -> None:
        """Replace target with attenuated source.

        Args:
            dst_fbo: Target framebuffer
            src: Source texture
            strength: Attenuation factor (1.0 = full copy, 0.0 = clear)
        """
        # Lazy init shader
        if not hasattr(FlowUtil, '_scale_shader'):
            from .shaders.Scale import Scale
            FlowUtil._scale_shader = Scale()
            FlowUtil._scale_shader.allocate()

        dst_fbo.swap()
        dst_fbo.begin()
        FlowUtil._scale_shader.use(src, strength)
        dst_fbo.end()

    @staticmethod
    def magnitude(dst_fbo: Fbo, src: Texture) -> None:
        """Compute vector magnitude from source texture.

        Computes length of RGBA vector and outputs to R channel.

        Args:
            dst_fbo: Destination FBO (typically R32F format)
            src: Source texture (any RGBA texture)
        """
        # Lazy init shader
        if not hasattr(FlowUtil, '_magnitude_shader'):
            from .shaders.Magnitude import Magnitude
            FlowUtil._magnitude_shader = Magnitude()
            FlowUtil._magnitude_shader.allocate()

        dst_fbo.begin()
        FlowUtil._magnitude_shader.use(src)
        dst_fbo.end()

    @staticmethod
    def normalize(dst_fbo: Fbo, src: Texture) -> None:
        """Normalize vectors to unit length.

        Converts all vectors to magnitude 1 while preserving direction.
        Zero vectors remain zero.

        Args:
            dst_fbo: Destination FBO
            src: Source texture (any RGBA texture)
        """
        # Lazy init shader
        if not hasattr(FlowUtil, '_normalize_shader'):
            from .shaders.Normalize import Normalize
            FlowUtil._normalize_shader = Normalize()
            FlowUtil._normalize_shader.allocate()

        dst_fbo.begin()
        FlowUtil._normalize_shader.use(src)
        dst_fbo.end()

    @staticmethod
    def get_num_channels(internal_format) -> int:
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
