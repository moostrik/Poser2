"""Gradient Flow: the 2D optical flow, as on main.

Computes the velocity field along the brightness gradient between two frames: hd_trio's flow.
Ported from ofxFlowTools ftOpticalFlow.h
"""
from OpenGL.GL import GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR, GL_R8, GL_RED, GL_RG16F, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glGenerateMipmap, glTexParameteri

from modules.gl import Fbo, Texture, Blit

from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil
from .shaders import Luminance, GradientFlowShader

from modules.utils import HotReloadMethods


class GradientFlowSettings(BaseSettings):
    offset = Field(3, min=1, max=10, description="Gradient sample offset in pixels")
    threshold = Field(0.0, min=0.0, max=0.2, description="Motion detection threshold")
    strength_x = Field(3.3, min=-10.0, max=10.0, description="X velocity multiplier (negative inverts)")
    strength_y = Field(3.3, min=-10.0, max=10.0, description="Y velocity multiplier (negative inverts)")
    boost = Field(0.0, min=-0.5, max=0.9, description="Power boost for small motions")


class GradientFlow(FlowBase):
    """Compute the gradient flow's velocity field from sequential frames.

    Uses input_fbo as SwapFbo to store current and previous frames.
    """

    def __init__(self, settings: GradientFlowSettings | None = None) -> None:
        super().__init__()

        # Define internal formats
        self._input_internal_format = GL_R8      # Luminance input (current/previous frames)
        self._output_internal_format = GL_RG16F  # Velocity output

        # Configuration with change notification
        self.settings: GradientFlowSettings = settings or GradientFlowSettings()
        self.settings.bind_all(lambda v: self._on_settings_changed())

        # State
        self._frame_count: int = 0  # 0=no frames, 1=first frame, 2+=can compute flow
        self._needs_update: bool = False

        # Shaders
        self._gradient_flow_shader: GradientFlowShader = GradientFlowShader()
        self._luminance_shader: Luminance = Luminance()

        # hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def velocity(self) -> Texture:
        return self._output

    @property
    def color_input(self) -> Texture:
        """Density input texture (luminance)."""
        return self._input_fbo.texture

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate optical flow layer."""

        super().allocate(width, height, output_width, output_height)

        for tex in [self._input_fbo.texture, self._input_fbo.back_texture]:
            glBindTexture(GL_TEXTURE_2D, tex.tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)  # Initial generation
        glBindTexture(GL_TEXTURE_2D, 0)

        self._gradient_flow_shader.allocate()
        self._luminance_shader.allocate()

        self._frame_count = 0
        self._needs_update = False

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        self._gradient_flow_shader.deallocate()
        self._luminance_shader.deallocate()

    def update(self, delta_time: float = 1.0) -> None:
        """Compute optical flow from current and previous frames.
        note :: delta_time parameter is unused but kept for consistency.
        """
        if not self._allocated or not self._needs_update:
            return

        self._needs_update = False

        # Need at least 2 frames to compute flow
        if self._frame_count < 2:
            FlowUtil.zero(self._output_fbo)
            return

        # Get current and previous frames from input_fbo using properties
        curr_frame = self._input_fbo.texture      # Current buffer (Fbo)
        prev_frame = self._input_fbo.back_texture # Previous buffer (Fbo)

        # Compute optical flow using settings values
        power = 1.0 - self.settings.boost

        self._gradient_flow_shader.reload()

        # Pass output_fbo directly since it's now a Fbo
        self._output_fbo.begin()
        self._gradient_flow_shader.use(
            prev_frame,
            curr_frame,
            offset=self.settings.offset,
            threshold=self.settings.threshold,
            strength_x=self.settings.strength_x,
            strength_y=self.settings.strength_y,
            power=power
        )
        self._output_fbo.end()

    def set_input(self, texture: Texture) -> None:
        """Set input frame texture.

        Args:
            texture: Input texture (will be converted to luminance)
        """
        if not self._allocated or not texture.allocated:
            return

        self._input_fbo.swap()

        if texture.format == GL_RED:
            # Input is already single-channel luminance
            self._input_fbo.begin()
            Blit.use(texture)
            self._input_fbo.end()

        else:
            # Convert to luminance using shader
            self._input_fbo.begin()
            self._luminance_shader.use(texture)
            self._input_fbo.end()

        # Generate mipmaps after rendering to FBO
        glBindTexture(GL_TEXTURE_2D, self._input_fbo.texture.tex_id)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._frame_count += 1
        self._needs_update = True

    def reset(self) -> None:
        """Reset optical flow state."""
        super().reset()
        self._frame_count = 0
        self._needs_update = False

    def _on_settings_changed(self) -> None:
        """Called when settings values change - trigger recompute if we have frames."""
        self._needs_update = True