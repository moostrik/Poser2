"""Optical Flow Layer.

Computes optical flow velocity field from input frames.
Ported from ofxFlowTools ftOpticalFlow.h
"""
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl.Fbo import Fbo
from modules.gl.Texture import Texture

from .. import FlowBase, FlowConfigBase, FlowUtil
from .shaders import Luminance, MergeRGB, OpticalFlow as OpticalFlowShader, OpticalFlowMM as OpticalFlowMMShader

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class OpticalFlowConfig(FlowConfigBase):
    offset: int = field(
        default=3,
        metadata={"min": 1, "max": 10, "label": "Offset", "description": "Gradient sample offset in pixels"}
    )
    threshold: float = field(
        default=0.1,
        metadata={"min": 0.0, "max": 0.2, "label": "Threshold", "description": "Motion detection threshold"}
    )
    strength_x: float = field(
        default=3.0,
        metadata={"min": -10.0, "max": 10.0, "label": "Strength X", "description": "X velocity multiplier (negative inverts)"}
    )
    strength_y: float = field(
        default=3.0,
        metadata={"min": -10.0, "max": 10.0, "label": "Strength Y", "description": "Y velocity multiplier (negative inverts)"}
    )
    boost: float = field(
        default=0.0,
        metadata={"min": -0.5, "max": 0.9, "label": "Boost", "description": "Power boost for small motions"}
    )


class OpticalFlow(FlowBase):
    """Compute optical flow velocity field from sequential frames.

    Uses input_fbo as SwapFbo to store current and previous frames.
    """

    def __init__(self, config: OpticalFlowConfig | None = None) -> None:
        super().__init__()

        # Define internal formats
        self._input_internal_format = GL_R8      # Luminance input (current/previous frames)
        self._output_internal_format = GL_RG32F  # Velocity output

        # Configuration with change notification
        self.config: OpticalFlowConfig = config or OpticalFlowConfig()
        self.config.add_listener(self._on_config_changed)

        # State
        self._frame_count: int = 0  # 0=no frames, 1=first frame, 2+=can compute flow
        self._needs_update: bool = False

        # Shaders
        self._optical_flow_shader: OpticalFlowShader = OpticalFlowShader()
        self._optical_flow_shader_mm: OpticalFlowMMShader = OpticalFlowMMShader()
        self._luminance_shader: Luminance = Luminance()
        self._merge_rgb_shader: MergeRGB = MergeRGB()

        hot_reload = HotReloadMethods(self.__class__, True, True)

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

        self._optical_flow_shader.allocate()
        self._optical_flow_shader_mm.allocate()
        self._luminance_shader.allocate()
        self._merge_rgb_shader.allocate()

        self._frame_count = 0
        self._needs_update = False

    def deallocate(self) -> None:
        """Release all resources."""
        super().deallocate()
        self._optical_flow_shader.deallocate()
        self._optical_flow_shader_mm.deallocate()
        self._luminance_shader.deallocate()
        self._merge_rgb_shader.deallocate()

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

        # Compute optical flow using config values
        power = 1.0 - self.config.boost

        self._optical_flow_shader.reload()
        self._optical_flow_shader_mm.reload()

        # Pass output_fbo directly since it's now a Fbo
        self._output_fbo.begin()
        self._optical_flow_shader_mm.use(
            prev_frame,
            curr_frame,
            offset=self.config.offset,
            threshold=self.config.threshold,
            strength_x=self.config.strength_x,
            strength_y=self.config.strength_y,
            power=power
        )
        self._output_fbo.end()

    def set_color(self, texture: Texture) -> None:
        """Set input frame texture.

        Args:
            texture: Input texture (will be converted to luminance)
        """
        if not self._allocated:
            return

        # Swap to next frame slot in input_fbo
        self._input_fbo.swap()
        self._merge_rgb_shader.reload()

        self._input_fbo.begin()
        # self._luminance_shader.use(texture)
        self._merge_rgb_shader.use(texture)
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

    def _on_config_changed(self) -> None:
        """Called when config values change - trigger recompute if we have frames."""
        self._needs_update = True