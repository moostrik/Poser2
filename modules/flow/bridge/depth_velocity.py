"""Depth Velocity.

What the image cannot measure: the player's motion toward the camera, from the depth image.
Frame to frame depth is mostly noise, so it is smoothed before it is differenced. Only the player
counts, and a pixel that jumps is an edge, not motion: a limb passing over a pixel is not z motion.
The change is measured like the flow's x/y, in view widths per frame: over the metres one view width
spans at the depth (`view_span` per metre). Both optical flows use it for their z, each turning it
into view widths per second times its own `strength_z`.
"""
from OpenGL.GL import GL_R16F, GL_R32F

from modules.gl import Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase
from ..shaders import GaussianBlur
from .kinetics import decay_weight
from .shaders import DepthSmooth, DepthThreshold, DepthVelocityShader


class DepthVelocitySettings(BaseSettings):
    """Configuration for the depth velocity."""
    smooth_time = Field(60.0, min=0.0, max=500.0, description="Smoothing of the depth before it is differenced (ms)")
    jump = Field(0.1, min=0.01, max=1.0, description="Depth change per frame above which a pixel is an edge (m)")
    blur_radius = Field(4.0, min=0.0, max=10.0, description="Gaussian blur radius in pixels")
    blur_steps = Field(2, min=0, max=8, description="Number of Gaussian blur passes")
    threshold = Field(0.5, min=0.0, max=5.0, step=0.05, description="Depth motion below this is noise, taken off the rest (px per frame)")


class DepthVelocity(FlowBase):
    """Signed motion toward the camera in view widths per frame (R16F), positive toward it.

    Pipeline, on each new flow frame:
        1. Smooth the depth → the input SwapFbo (R32F, metres, 0 off the player)
        2. Difference it with the last one, over the metres of a view width there → output_fbo (R16F)
        3. Blur it, and take the noise floor off
    """

    def __init__(self, settings: DepthVelocitySettings | None = None) -> None:
        super().__init__()

        self.settings: DepthVelocitySettings = settings or DepthVelocitySettings()

        self._input_internal_format = GL_R32F
        self._output_internal_format = GL_R16F

        self._smooth_shader: DepthSmooth = DepthSmooth()
        self._velocity_shader: DepthVelocityShader = DepthVelocityShader()
        self._blur_shader: GaussianBlur = GaussianBlur()
        self._threshold_shader: DepthThreshold = DepthThreshold()

    @property
    def depth_velocity(self) -> Texture:
        """Motion toward the camera in view widths per frame (R16F)."""
        return self._output

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        super().allocate(width, height, output_width, output_height)
        self._smooth_shader.allocate()
        self._velocity_shader.allocate()
        self._blur_shader.allocate()
        self._threshold_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._smooth_shader.deallocate()
        self._velocity_shader.deallocate()
        self._blur_shader.deallocate()
        self._threshold_shader.deallocate()

    def update_depth(self, depth: Texture, delta_time: float, view_span: float, displacement: Texture | None = None) -> None:
        """Advance on a new flow frame: smooth, then difference.

        Args:
            depth: Metres in red, mask in alpha
            delta_time: Seconds since the previous flow frame; sets the smoothing's weight
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            displacement: How far each pixel moved since then (uv); when given, the smoothing and the
                difference follow it, so depth is measured along the motion (Lucas–Kanade flow)
        """
        if not self._allocated:
            return
        s: DepthVelocitySettings = self.settings

        self._input_fbo.swap()
        self._input_fbo.begin()
        self._smooth_shader.use(depth, self._input_fbo.back_texture, decay_weight(delta_time, s.smooth_time), s.jump, displacement)
        self._input_fbo.end()

        self._output_fbo.begin()
        self._velocity_shader.use(self._input_fbo.texture, self._input_fbo.back_texture, view_span, s.jump, displacement)
        self._output_fbo.end()

        if s.blur_radius > 0.0:
            for _ in range(s.blur_steps):
                for horizontal in (True, False):
                    self._output_fbo.swap()
                    self._output_fbo.begin()
                    self._blur_shader.use(self._output_fbo.back_texture, s.blur_radius, horizontal)
                    self._output_fbo.end()

        if s.threshold > 0.0:
            self._output_fbo.swap()
            self._output_fbo.begin()
            self._threshold_shader.use(self._output_fbo.back_texture, s.threshold / max(self._output_fbo.width, 1))    # px as view widths
            self._output_fbo.end()
