"""Lucas–Kanade Flow 3D: the x/y solve (`LucasKanadeFlow`) and the depth velocity (`DepthVelocity`) composed into
one 3D velocity in image widths per second times `x_gain`, `y_gain`, `z_gain`, gated by the mask with `use_mask`
and z by the depth mask; beside it `z_position`, where the body is now in the player's depth range, 0 front to 1
back, times its weight in red and the weight in green (RG16F). How and why: modules/flow/MOTION.md, "The 3D
velocity" and "depth_velocity (z)".

`filter_displacement`, `displacement_velocity` and `z_position` below are the numpy references of the shaders.
"""

import math

import numpy as np
from OpenGL.GL import GL_RG16F, GL_RG32F, GL_RGBA16F, GL_RGBA32F

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field, Group, Widget
from .. import FlowUtil, Port, TextureKind, prefixed
from ..finite_guard import FiniteGuard, GuardSlot
from .depth_velocity import DepthVelocity, DepthVelocitySettings
from .solve_capture import SolveCapture
from .lucas_kanade_flow import LucasKanadeFlow, LucasKanadeFlowSettings, _bilinear
from .shaders import ComposeVelocity, DisplacementFilter, DisplacementThreshold, ZPosition

MIN_DELTA_TIME: float = 1e-3        # s: an interval at or below this is not a frame's
FALLBACK_DELTA_TIME: float = 1.0 / 30.0     # s: the interval taken in its place


class LucasKanadeFlow3DSettings(BaseSettings):
    """Configuration for the 3D Lucas–Kanade flow: the mask and the axes, its x/y solve in `xy` and its depth velocity in `z`."""
    use_mask:       Field[bool]  = Field(True, newline=True, width=48, description="Only the player moves the fluid (off: whole image)")
    x_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, image widths per second")
    y_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, image widths per second")
    z_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, on the x/y scale")

    xy             = Group(LucasKanadeFlowSettings)
    z              = Group(DepthVelocitySettings)


class LucasKanadeFlow3D:
    """The 3D velocity (RGBA16F, xyz) between the newest two frames of its input."""

    def __init__(self, settings: LucasKanadeFlow3DSettings) -> None:
        self.settings: LucasKanadeFlow3DSettings = settings
        self._flow: LucasKanadeFlow = LucasKanadeFlow(settings.xy)
        self._depth_velocity: DepthVelocity = DepthVelocity()
        # The outputs' non-finite values, set to 0 where each is written, counted and reported: the velocity, and the
        # two memories a reset clears, the x/y filter's (displacement) and z's (depth_velocity)
        self._guard: FiniteGuard = FiniteGuard('motion', ('velocity', 'displacement', 'depth_velocity'))
        # The solve's inputs kept when it comes out wrong, with POSER_SOLVE_CAPTURE set (a debugging tool)
        self._capture: SolveCapture | None = SolveCapture.from_environment()
        self._mask: Texture | None = None

        self._filtered: SwapFbo = SwapFbo()         # with z_smooth: the filter, its displacement and position error
        self._followed: Fbo = Fbo()                 # the filtered displacement with the noise floor taken off
        self._following: bool = False
        self._velocity: Fbo = Fbo()
        self._z_position: Fbo = Fbo()               # where the body is now in the depth range, red over green
        self._z_position_shader: ZPosition = ZPosition()
        self._threshold_shader: DisplacementThreshold = DisplacementThreshold()
        self._filter_shader: DisplacementFilter = DisplacementFilter()
        self._compose_shader: ComposeVelocity = ComposeVelocity()

    @property
    def luminance(self) -> Texture:
        """The newest luminance frame."""
        return self._flow.luminance

    @property
    def velocity(self) -> Texture:
        """x, y and z in image widths per second, each times its gain (RGBA16F)."""
        return self._velocity

    @property
    def z_position(self) -> Texture:
        """Where the body is now in the depth range, 0 front to 1 back, times its weight in red, the weight in
        green (RG16F); 0 without depth."""
        return self._z_position

    @property
    def depth_velocity(self) -> DepthVelocity:
        """Its z, before `z_gain` and the mask."""
        return self._depth_velocity

    @property
    def ports(self) -> dict[str, Port]:
        """Its two parts under their settings groups: the x/y solve with its smoothing, and the z.
        The composed velocity is the motion stage's output."""
        return {**prefixed('xy', {**self._flow.ports,
                                  'filtered': Port(self._filtered, TextureKind.VELOCITY, held=2),     # BA the position error, state
                                  'followed': Port(self._followed, TextureKind.VELOCITY)}),
                **prefixed('z', self._depth_velocity.ports),
                'z_position': Port(self._z_position, TextureKind.Z_POSITION)}

    def allocate(self, width: int, height: int) -> None:
        self._flow.allocate(width, height)
        self._depth_velocity.allocate(width, height)
        self._filtered.allocate(width, height, GL_RGBA32F)
        self._followed.allocate(width, height, GL_RG32F)
        self._velocity.allocate(width, height, GL_RGBA16F)
        self._z_position.allocate(width, height, GL_RG16F)
        self._z_position_shader.allocate()
        self._threshold_shader.allocate()
        self._filter_shader.allocate()
        self._compose_shader.allocate()
        self._guard.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._flow.deallocate()
        self._depth_velocity.deallocate()
        self._filtered.deallocate()
        self._followed.deallocate()
        self._velocity.deallocate()
        self._z_position.deallocate()
        self._z_position_shader.deallocate()
        self._threshold_shader.deallocate()
        self._filter_shader.deallocate()
        self._compose_shader.deallocate()
        self._guard.deallocate()

    def reset(self) -> None:
        self._flow.reset()
        self._depth_velocity.reset()
        self._filtered.clear_all()
        self._followed.clear()
        self._velocity.clear()
        self._z_position.clear()

    def _slot(self, output: str, guarded: bool) -> GuardSlot | None:
        """The output's slot in the motion's guard while `guarded`, else None."""
        return self._guard.slot(output) if guarded else None

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._flow.set_input(texture)

    def set_mask(self, texture: Texture | None) -> None:
        """The player's mask (red, 0 to 1) the velocity is multiplied by with `use_mask`, or None for no mask."""
        self._mask = texture

    def update(self, depth: Texture | None, delta_time: float, view_span: float, depth_step: tuple[float, float],
               guarded: bool = False, depth_range: tuple[float, float] = (0.0, 0.0)) -> None:
        """Measure between the newest two frames.

        Args:
            depth: Metres in red; None without depth (z is 0)
            delta_time: Seconds since the previous frame
            view_span: Metres one view width spans per metre of depth, for z
            depth_step: uv between two depth samples in the flow's view, for z
            guarded: Set the outputs' non-finite values to 0, count and log them (`guard`)
            depth_range: The player's depth range, near and far (m), for `z_position`
        """
        if not self._flow.ready:
            FlowUtil.zero(self._velocity)
            return
        s: LucasKanadeFlow3DSettings = self.settings
        xy_s: LucasKanadeFlowSettings = s.xy
        z_s: DepthVelocitySettings = s.z

        captured = self._capture.before(self._flow.previous_luminance, self._flow.luminance, xy_s.levels, depth) if self._capture else None
        self._flow.update()
        solved: Texture = self._flow.solved
        if self._capture is not None:
            self._capture.after(captured, solved, {'levels': xy_s.levels, 'window': int(xy_s.window), 'damping': xy_s.damping})

        if depth is not None:
            self._depth_velocity.update_depth(depth, delta_time, view_span, depth_step, z_s.window, z_s.smooth, z_s.xy_halving,
                                              z_s.xy_rest, z_s.threshold, z_s.edge, z_s.dead_zone, solved, z_s.max_speed,
                                              self._slot('depth_velocity', guarded))

        xy: Texture = self._flow.displacement
        if xy_s.z_smooth:
            if not self._following:
                self._filtered.clear_all()                  # from no motion, not from when it was last on
                self._following = True
            self._filtered.swap()
            self._filtered.begin()
            self._filter_shader.use(solved, self._filtered.back, solved, z_s.smooth, z_s.xy_rest, z_s.xy_halving,
                                    max(delta_time, 1e-6), self._slot('displacement', guarded))
            self._filtered.end()
            self._followed.begin()
            self._threshold_shader.use(self._filtered, xy_s.threshold)
            self._followed.end()
            xy = self._followed
        else:
            self._following = False

        # uv per frame as image widths per second: v is uv of the height, so a square motion stays square;
        # z is in view widths per frame already
        # an interval that is not a frame's (0, negative, not finite) would scale every pixel at once
        per_second: float = 1.0 / delta_time if math.isfinite(delta_time) and delta_time > MIN_DELTA_TIME else 1.0 / FALLBACK_DELTA_TIME
        aspect: float = self._velocity.height / max(self._velocity.width, 1)
        mask: Texture | None = self._mask if s.use_mask and self._mask is not None and self._mask.allocated else None
        self._velocity.begin()
        self._compose_shader.use(xy, self._depth_velocity.depth_velocity if depth is not None else None,
                                 (per_second * s.x_gain, per_second * aspect * s.y_gain, per_second * s.z_gain), mask,
                                 self._slot('velocity', guarded))
        self._velocity.end()

        if depth is not None:
            self._z_position.begin()
            self._z_position_shader.use(self._depth_velocity.difference, depth_range[0], depth_range[1], mask)
            self._z_position.end()
        else:
            self._z_position.clear()
        self._guard.update(delta_time)


# ── numpy reference ─────────────────────────────────────────────────────

def z_position(disparity: np.ndarray, near: float, far: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Where the body is now in the depth range (..., 2): the z position, (depth - near) / (far - near) held to
    0..1, times its weight, and the weight: the mask's (1 without), 0 without depth (1 / `disparity`, m). Mirrors
    z_position.frag."""
    depth: np.ndarray = np.where(disparity > 1e-6, 1.0 / np.maximum(disparity, 1e-6), 0.0)
    weight: np.ndarray = ((depth > 0.0) & (far > near)).astype(np.float64)
    if mask is not None:
        weight = weight * mask
    z: np.ndarray = np.where(weight > 0.0, np.clip((depth - near) / max(far - near, 1e-9), 0.0, 1.0), 0.0)
    return np.stack((z * weight, weight), axis=-1)


def filter_displacement(displacement: np.ndarray, state: np.ndarray, carry: np.ndarray, alpha: float) -> np.ndarray:
    """The filter x/y goes through with xy.z_smooth, one frame on (H, W, 4): the filtered
    displacement (x, y, pixels per frame) and the position error (x, y, pixels). Mirrors
    displacement_filter.frag.

    The depth velocity's alpha–beta filter, on positions that step by `displacement` (H, W, 2, pixels
    per frame) each frame: the state (zeros to start) is carried from where each pixel was `carry`
    back, linear between texels, its displacement predicts the step, and the residual corrects it by
    `alpha` and alpha² / (2 − alpha), as a full window corrects z.
    """
    h, w = displacement.shape[:2]
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = columns - carry[..., 0], rows - carry[..., 1]
    carried: np.ndarray = np.stack([_bilinear(state[..., channel], x, y) for channel in range(4)], axis=-1)
    beta: float = alpha * alpha / (2.0 - alpha)
    residual: np.ndarray = displacement - carried[..., 2:] - carried[..., :2]
    return np.concatenate((carried[..., :2] + beta * residual, -(1.0 - alpha) * residual), axis=-1)


def displacement_velocity(displacement: np.ndarray, dt: float, width: int, height: int, strength: tuple[float, float]) -> np.ndarray:
    """A displacement in uv per frame (..., 2) as a velocity in image widths per second, times `strength`.

    v is uv of the height, so it is taken to widths first: a square motion stays square.
    """
    return displacement * np.array([1.0, height / width]) / dt * np.array(strength)
