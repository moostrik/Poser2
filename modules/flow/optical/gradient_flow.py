"""Gradient Flow: the original optical flow.

Per pixel, the brightness change along the brightness gradient between two frames: the motion along
the gradient, as a strength rather than a distance. One equation per pixel, so only that component.
x/y are scaled to the flow's velocity by `strength_x`/`strength_y`, then `threshold` comes off their
length; z is the depth change at the pixel (`DepthVelocity`) in view widths per second times
`strength_z`. With `use_mask`, the velocity is multiplied by the mask.
Ported from ofxFlowTools ftOpticalFlow.h
"""
from OpenGL.GL import GL_RG16F, GL_RGBA16F

from modules.gl import Fbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowUtil
from ..bridge import DepthVelocity
from .luminance_frames import LuminanceFrames
from .shaders import ComposeVelocity, GradientFlowShader


class GradientFlowSettings(BaseSettings):
    """Configuration for the gradient flow."""
    offset = Field(3, min=1, max=10, description="Gradient sample offset in pixels")
    threshold = Field(0.0, min=0.0, max=0.2, description="Motion detection threshold")
    strength_x = Field(3.3, min=-10.0, max=10.0, description="X velocity multiplier (negative inverts)")
    strength_y = Field(3.3, min=-10.0, max=10.0, description="Y velocity multiplier (negative inverts)")
    strength_z = Field(0.0, min=0.0, max=10.0, description="Velocity per view width per second toward the camera; 0 is no depth")
    use_mask = Field(False, description="Multiply the velocity by the mask")


class GradientFlow:
    """The 3D velocity (RGBA16F, xyz) between the newest two frames of its input."""

    def __init__(self, settings: GradientFlowSettings, depth_velocity: DepthVelocity) -> None:
        self.settings: GradientFlowSettings = settings
        self._depth_velocity: DepthVelocity = depth_velocity     # shared with the other flow; only the active one advances it
        self._mask: Texture | None = None

        self._frames: LuminanceFrames = LuminanceFrames()
        self._xy: Fbo = Fbo()
        self._velocity: Fbo = Fbo()
        self._shader: GradientFlowShader = GradientFlowShader()
        self._compose_shader: ComposeVelocity = ComposeVelocity()

    @property
    def luminance(self) -> Texture:
        """The newest luminance frame."""
        return self._frames.luminance

    @property
    def velocity(self) -> Texture:
        """x/y and z (RGBA16F)."""
        return self._velocity

    def allocate(self, width: int, height: int) -> None:
        self._frames.allocate(width, height)
        self._xy.allocate(width, height, GL_RG16F)
        self._velocity.allocate(width, height, GL_RGBA16F)
        self._shader.allocate()
        self._compose_shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._frames.deallocate()
        self._xy.deallocate()
        self._velocity.deallocate()
        self._shader.deallocate()
        self._compose_shader.deallocate()

    def reset(self) -> None:
        self._frames.reset()
        FlowUtil.zero(self._xy)
        FlowUtil.zero(self._velocity)

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._frames.set_input(texture)

    def set_mask(self, texture: Texture | None) -> None:
        """The player's mask (red, 0 to 1) the velocity is multiplied by with `use_mask`, or None for no mask."""
        self._mask = texture

    def update(self, depth: Texture | None, delta_time: float, view_span: float) -> None:
        """Measure between the newest two frames.

        Args:
            depth: Metres in red, mask in alpha; None without depth (z is 0)
            delta_time: Seconds since the previous frame
            view_span: Metres one view width spans per metre of depth, for z
        """
        if not self._frames.ready:
            FlowUtil.zero(self._velocity)
            return
        s: GradientFlowSettings = self.settings

        self._xy.begin()
        self._shader.use(self._frames.previous_luminance, self._frames.luminance, s.offset, s.threshold, s.strength_x, s.strength_y)
        self._xy.end()

        if depth is not None:
            self._depth_velocity.update_depth(depth, delta_time, view_span)

        # z in view widths per frame as view widths per second
        per_second: float = 1.0 / max(delta_time, 1e-6)
        mask: Texture | None = self._mask if s.use_mask and self._mask is not None and self._mask.allocated else None
        self._velocity.begin()
        self._compose_shader.use(self._xy, self._depth_velocity.depth_velocity if depth is not None else None, (1.0, 1.0, s.strength_z * per_second), mask)
        self._velocity.end()
