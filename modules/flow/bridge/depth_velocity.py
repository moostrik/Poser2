"""Depth Velocity.

What optical flow cannot measure: the player's velocity toward the camera, from the depth image.
Frame to frame depth is mostly noise, so it is smoothed before it is differenced. Only the player
counts, and a pixel that jumps is an edge, not motion: a limb passing over a pixel is not z motion.
"""
from OpenGL.GL import GL_R16F, GL_R32F

from modules.gl import Fbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil
from ..shaders import GaussianBlur
from .kinetics import decay_weight
from .shaders import DepthDeadZone, DepthSmooth, DepthVelocityShader, Speed3D


class DepthVelocitySettings(BaseSettings):
    """Configuration for the depth velocity."""
    scale = Field(0.0, min=0.0, max=10.0, description="Converts m/s toward the camera to simulation velocity; 0 is off")
    smooth_time = Field(60.0, min=0.0, max=500.0, description="Smoothing of the depth before it is differenced (ms)")
    jump = Field(0.1, min=0.01, max=1.0, description="Depth change per frame above which a pixel is an edge (m)")
    blur_radius = Field(4.0, min=0.0, max=10.0, description="Gaussian blur radius in pixels")
    blur_steps = Field(2, min=0, max=8, description="Number of Gaussian blur passes")
    dead_zone = Field(0.1, min=0.0, max=1.0, description="Velocity toward the camera taken off as noise floor (m/s)")


class DepthVelocity(FlowBase):
    """Signed velocity toward the camera (R16F), positive toward it.

    Pipeline, on each new optical flow frame:
        1. Smooth the depth → the input SwapFbo (R32F, metres, 0 off the player)
        2. Difference it with the last one → output_fbo (R16F)
        3. Blur it, and take the noise floor off

    `process_speed` gives the bridges the 3D speed that the density and temperature bridges read.
    """

    def __init__(self, settings: DepthVelocitySettings | None = None) -> None:
        super().__init__()

        self.settings: DepthVelocitySettings = settings or DepthVelocitySettings()

        self._input_internal_format = GL_R32F
        self._output_internal_format = GL_R16F

        self._smooth_shader: DepthSmooth = DepthSmooth()
        self._velocity_shader: DepthVelocityShader = DepthVelocityShader()
        self._blur_shader: GaussianBlur = GaussianBlur()
        self._dead_zone_shader: DepthDeadZone = DepthDeadZone()
        self._speed_shader: Speed3D = Speed3D()

        self._speed_fbo: Fbo = Fbo()

    @property
    def depth_velocity(self) -> Texture:
        """Velocity toward the camera (R16F)."""
        return self._output

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        super().allocate(width, height, output_width, output_height)
        self._speed_fbo.allocate(width, height, GL_R16F)
        self._smooth_shader.allocate()
        self._velocity_shader.allocate()
        self._blur_shader.allocate()
        self._dead_zone_shader.allocate()
        self._speed_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._speed_fbo.deallocate()
        self._smooth_shader.deallocate()
        self._velocity_shader.deallocate()
        self._blur_shader.deallocate()
        self._dead_zone_shader.deallocate()
        self._speed_shader.deallocate()

    def process_speed(self, velocity: Texture, depth_velocity: Texture) -> Texture:
        """The speed of a bridge's (x, y) and z (R16F): its length reads like a velocity's."""
        self._speed_fbo.begin()
        self._speed_shader.use(velocity, depth_velocity)
        self._speed_fbo.end()
        return self._speed_fbo

    def update_depth(self, depth: Texture, delta_time: float, displacement: Texture | None = None) -> None:
        """Advance on a new optical flow frame.

        Args:
            depth: Metres in red, mask in alpha
            delta_time: Seconds since the previous optical flow frame
            displacement: How far each pixel moved since then (uv); when given, the smoothing and the
                difference follow it, so depth is measured along the motion (scene flow)
        """
        if not self._allocated:
            return

        s: DepthVelocitySettings = self.settings

        self._input_fbo.swap()
        self._input_fbo.begin()
        self._smooth_shader.use(depth, self._input_fbo.back_texture, decay_weight(delta_time, s.smooth_time), s.jump, displacement)
        self._input_fbo.end()

        self._output_fbo.begin()
        self._velocity_shader.use(self._input_fbo.texture, self._input_fbo.back_texture, delta_time, s.jump, s.scale, displacement)
        self._output_fbo.end()

        if s.blur_radius > 0.0:
            for _ in range(s.blur_steps):
                for horizontal in (True, False):
                    self._output_fbo.swap()
                    self._output_fbo.begin()
                    self._blur_shader.use(self._output_fbo.back_texture, s.blur_radius, horizontal)
                    self._output_fbo.end()

        if s.dead_zone > 0.0:
            self._output_fbo.swap()
            self._output_fbo.begin()
            self._dead_zone_shader.use(self._output_fbo.back_texture, s.dead_zone * s.scale)
            self._output_fbo.end()
