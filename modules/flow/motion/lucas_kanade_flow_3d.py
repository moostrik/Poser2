"""Lucas–Kanade Flow 3D: the x/y solve (`LucasKanadeFlow`) and the depth velocity (`DepthVelocity`)
composed into one 3D velocity.

The displacement (uv per frame) stays inside: `solve.threshold` (px per frame) comes off its length,
as below it is the solve's jitter, not motion. z comes from its own `DepthVelocity`: a depth track carried
along the displacement, so a surface sliding sideways over a slope reads no motion toward the camera,
corrected each frame by the mean over `depth_velocity.window` depth samples and smoothed over
`depth_velocity.smooth`; `depth_velocity.threshold` comes off it. The depth change comes in view
widths per frame like the displacement, so all three become view widths per second the same way. Out
comes one 3D velocity in image widths per second times `x_gain`, `y_gain` and `z_gain` (1 = true speed,
0 = that axis off, negative = inverted). With `use_mask`, the velocity is multiplied by the mask.

z lags its motion by the time it is smoothed over. With `depth_velocity.xy_smooth`, x/y goes through the
same filter over `depth_velocity.smooth` before its threshold, carried along the displacement as z is, so
both lag alike. A displacement is a step of position, so the filter is the same one: with a steady frame
interval and a full z window, x/y answers a change of speed exactly as z does. The tracks are
carried by the displacement as solved, before `solve.threshold`: a slide too slow for the output
still moves the tracks with it, or on a slope it would read as z.

The settings in flow pixels go with the flow's resolution: the solve's, and `depth_velocity.threshold`.
Those in metres and milliseconds do not.

`filter_displacement` and `displacement_velocity` below are the numpy references of the shaders.
"""

import math

import numpy as np
from OpenGL.GL import GL_RG32F, GL_RGBA16F, GL_RGBA32F

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field, Group, Widget
from .. import FlowUtil
from .depth_velocity import DepthVelocity, DepthVelocitySettings, track_weights
from .lucas_kanade_flow import LucasKanadeFlow, LucasKanadeFlowSettings, _bilinear
from .shaders import ComposeVelocity, DisplacementFilter, DisplacementThreshold

MIN_DELTA_TIME: float = 1e-3        # s: an interval at or below this is not a frame's
FALLBACK_DELTA_TIME: float = 1.0 / 30.0     # s: the interval taken in its place


class LucasKanadeFlow3DSettings(BaseSettings):
    """Configuration for the 3D Lucas–Kanade flow: the mask and the axes, its x/y solve and its depth velocity."""
    use_mask:       Field[bool]  = Field(True, width=48, description="Only the player moves the fluid (off: whole image)")
    x_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, image widths per second")
    y_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, image widths per second")
    z_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, step=0.05, widget=Widget.knob, description="1 = true speed, on the x/y scale")

    solve          = Group(LucasKanadeFlowSettings)
    depth_velocity = Group(DepthVelocitySettings)


class LucasKanadeFlow3D:
    """The 3D velocity (RGBA16F, xyz) between the newest two frames of its input."""

    def __init__(self, settings: LucasKanadeFlow3DSettings) -> None:
        self.settings: LucasKanadeFlow3DSettings = settings
        self._flow: LucasKanadeFlow = LucasKanadeFlow(settings.solve)
        self._depth_velocity: DepthVelocity = DepthVelocity()
        self._mask: Texture | None = None

        self._filtered: SwapFbo = SwapFbo()         # with xy_smooth: the filter, its displacement and position error
        self._followed: Fbo = Fbo()                 # the filtered displacement with the noise floor taken off
        self._following: bool = False
        self._velocity: Fbo = Fbo()
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
    def depth_velocity(self) -> DepthVelocity:
        """Its z, before `z_gain` and the mask."""
        return self._depth_velocity

    def allocate(self, width: int, height: int) -> None:
        self._flow.allocate(width, height)
        self._depth_velocity.allocate(width, height)
        self._filtered.allocate(width, height, GL_RGBA32F)
        self._followed.allocate(width, height, GL_RG32F)
        self._velocity.allocate(width, height, GL_RGBA16F)
        self._threshold_shader.allocate()
        self._filter_shader.allocate()
        self._compose_shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._flow.deallocate()
        self._depth_velocity.deallocate()
        self._filtered.deallocate()
        self._followed.deallocate()
        self._velocity.deallocate()
        self._threshold_shader.deallocate()
        self._filter_shader.deallocate()
        self._compose_shader.deallocate()

    def reset(self) -> None:
        self._flow.reset()
        self._depth_velocity.reset()
        self._filtered.clear_all()
        self._followed.clear()
        self._velocity.clear()

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._flow.set_input(texture)

    def set_mask(self, texture: Texture | None) -> None:
        """The player's mask (red, 0 to 1) the velocity is multiplied by with `use_mask`, or None for no mask."""
        self._mask = texture

    def update(self, depth: Texture | None, delta_time: float, view_span: float, depth_step: tuple[float, float]) -> None:
        """Measure between the newest two frames.

        Args:
            depth: Metres in red; None without depth (z is 0)
            delta_time: Seconds since the previous frame
            view_span: Metres one view width spans per metre of depth, for z
            depth_step: uv between two depth samples in the flow's view, for z
        """
        if not self._flow.ready:
            FlowUtil.zero(self._velocity)
            return
        s: LucasKanadeFlow3DSettings = self.settings
        xy_s: LucasKanadeFlowSettings = s.solve
        z_s: DepthVelocitySettings = s.depth_velocity

        self._flow.update(depth, z_s.tolerance)
        solved: Texture = self._flow.solved

        if depth is not None:
            self._depth_velocity.update_depth(depth, delta_time, view_span, depth_step, z_s.window, z_s.smooth,
                                              z_s.threshold, z_s.tolerance, z_s.dead_zone, z_s.catch_up, xy_s.threshold,
                                              solved, z_s.edge)

        xy: Texture = self._flow.displacement
        if z_s.xy_smooth:
            if not self._following:
                self._filtered.clear_all()                  # from no motion, not from when it was last on
                self._following = True
            alpha, _ = track_weights(delta_time, z_s.smooth, 1.0)
            self._filtered.swap()
            self._filtered.begin()
            self._filter_shader.use(solved, self._filtered.back_texture, solved, float(alpha))
            self._filtered.end()
            self._followed.begin()
            self._threshold_shader.use(self._filtered.texture, xy_s.threshold)
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
                                 (per_second * s.x_gain, per_second * aspect * s.y_gain, per_second * s.z_gain), mask)
        self._velocity.end()


# ── numpy reference ─────────────────────────────────────────────────────

def filter_displacement(displacement: np.ndarray, state: np.ndarray, carry: np.ndarray, alpha: float) -> np.ndarray:
    """The filter x/y goes through with depth_velocity.xy_smooth, one frame on (H, W, 4): the filtered
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
