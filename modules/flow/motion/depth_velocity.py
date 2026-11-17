"""Depth Velocity: the player's motion toward the camera, z, from the depth image, in view widths per frame:
per pixel a disparity and its rate, carried along the x/y displacement and corrected by each measurement
(an alpha–beta filter), with a dead zone, a catch-up and a step filter before it. How and why:
modules/flow/MOTION.md, "z: the depth velocity".

`filter_depth_steps` and `depth_threshold` below are the numpy references of their shaders.
"""
import math

import numpy as np
from OpenGL.GL import GL_R16F, GL_R32F, GL_RG32F, GL_RGBA32F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field
from .. import FlowBase, FlowUtil, Port, TextureKind
from .patch import Patch
from ..finite_guard import GuardSlot
from .shaders import DepthStepFilter, DepthThreshold, DepthVelocityShader


class DepthVelocitySettings(BaseSettings):
    """The z part: the depth velocity, a depth track carried along the displacement."""
    edge:           Field[float] = Field(0.1, min=0.05, max=0.3, step=0.01, description="Edge depth step, m (lower: cleaner edges, fewer fingers)")
    window:         Field[Patch] = Field(Patch.SEVEN, min=Patch.ONE, max=Patch.NINE, description="Depth averaging size (larger: less noise, small parts blur)")
    tolerance:      Field[float] = Field(0.2, min=0.05, max=0.5, step=0.01, description="New-surface depth jump, m (higher: fuller pushes, fewer splits)")
    smooth:         Field[float] = Field(50.0, min=0.0, max=500.0, step=1.0, newline=True, description="Smoothing time of z in ms, 0 = two-frame change (higher: less noise, more lag)")
    xy_smooth:      Field[bool]  = Field(False, width=48, description="x/y smoothed over smooth, so both lag alike (on: less noise, more lag)")
    catch_up:       Field[float] = Field(8.0, min=0.0, max=50.0, step=1.0, description="Shorter smoothing during a fast push, in ms, 0 = off (lower: quicker pushes)")
    dead_zone:      Field[float] = Field(0.02, min=0.0, max=0.05, step=0.001, newline=True, description="Depth drift ignored before z counts as motion, in m (higher: less drift, slower start)")
    max_speed:      Field[float] = Field(6.0, min=0.0, max=20.0, step=0.5, description="The fastest a body part moves in depth, m/s; a faster track reads another surface (0 = off)")
    threshold:      Field[float] = Field(1.0, min=0.0, max=5.0, step=0.05, description="z below this is noise, in px/frame (higher: less noise, less slow z)")


def track_weights(delta_time: float, smooth: float, share: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """The alpha–beta filter's weights: alpha on the depth, beta on the rate, for a measurement that
    holds `share` of its window. Alpha decays over `smooth` (ms), 1 at 0; beta is alpha² / (2 − alpha),
    critically damped, so the rate settles without overshoot."""
    alpha: float = 1.0 - math.exp(-delta_time * 1000.0 / smooth) if smooth > 0.0 else 1.0
    weighted = alpha * share
    return weighted, weighted * weighted / (2.0 - weighted)


class DepthVelocity(FlowBase):
    """Signed motion toward the camera in view widths per frame (R16F), positive toward it.

    Pipeline, on each new flow frame:
        1. Take the depth → the raw Fbo (R32F, metres, at the flow's size, nearest), and without
           its edges → the input SwapFbo
        2. Correct the track carried from the last one → the estimate SwapFbo (RGBA32F: z, disparity, rate, anchor),
           and in a second run of the same shader its catch-up → the catch-up SwapFbo (RG32F)
        3. Take the noise floor off z → output_fbo (R16F)
    """

    def __init__(self) -> None:
        super().__init__()

        self._input_internal_format = GL_R32F
        self._output_internal_format = GL_R16F
        self._input_kind = TextureKind.SCALAR
        self._output_kind = TextureKind.SCALAR

        self._step_filter_shader: DepthStepFilter = DepthStepFilter()
        self._velocity_shader: DepthVelocityShader = DepthVelocityShader()
        self._threshold_shader: DepthThreshold = DepthThreshold()

        self._raw: Fbo = Fbo()                   # the depth as it came, before its edges are taken off
        self._estimate: SwapFbo = SwapFbo()
        self._catch_up: SwapFbo = SwapFbo()      # the Emas of the miss of the prediction, signed and its size (RG32F, m)

    @property
    def depth_velocity(self) -> Texture:
        """Motion toward the camera in view widths per frame (R16F)."""
        return self._output

    @property
    def difference(self) -> Texture:
        """The estimate (RGBA32F): z before the threshold in red (view widths per frame), its disparity
        (1/m) in green, the disparity's rate (1/m per second) in blue and the dead zone's anchor (m) in alpha."""
        return self._estimate

    @property
    def ports(self) -> dict[str, Port]:
        """`input` is the depth with its edges taken off, `raw` as it came."""
        return {'raw': Port(self._raw, TextureKind.SCALAR), **super().ports,
                'estimate': Port(self._estimate, TextureKind.SCALAR),
                'catch_up': Port(self._catch_up, TextureKind.SCALAR)}

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        super().allocate(width, height, output_width, output_height)
        self._raw.allocate(width, height, GL_R32F)
        self._estimate.allocate(width, height, GL_RGBA32F)
        self._estimate.clear_all()
        self._catch_up.allocate(width, height, GL_RG32F)
        self._catch_up.clear_all()
        self._step_filter_shader.allocate()
        self._velocity_shader.allocate()
        self._threshold_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._raw.deallocate()
        self._estimate.deallocate()
        self._catch_up.deallocate()
        self._step_filter_shader.deallocate()
        self._velocity_shader.deallocate()
        self._threshold_shader.deallocate()

    def reset(self) -> None:
        super().reset()
        self._estimate.clear_all()
        self._catch_up.clear_all()

    def update_depth(self, depth: Texture, delta_time: float, view_span: float, depth_step: tuple[float, float], window: int,
                     smooth: float, threshold: float, tolerance: float, dead_zone: float, catch_up_smooth: float, motion_floor: float,
                     displacement: Texture | None = None, edge: float | None = None, max_speed: float = 0.0,
                     guard: GuardSlot | None = None) -> None:
        """Advance on a new flow frame: correct the track by the new depth, then take the noise floor off.

        Args:
            depth: Metres in red, 0 where there is none
            delta_time: Seconds since the last depth
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            depth_step: uv between two depth samples in the flow's view; 0 takes one flow texel
            window: Radius in depth samples
            smooth: Time over which the track averages (ms); 0 is the change of two frames
            threshold: Depth motion below this is noise, taken off the rest (px per frame)
            tolerance: Depth further than this is another surface: not averaged in, the track starts over (m)
            dead_zone: Depth the track wanders within its anchor without z (m); 0 is none
            catch_up_smooth: Time over which the track follows while it catches up a motion (ms); 0 is never
            motion_floor: Displacement the image must move by for the track to catch up (px per frame)
            displacement: How far each pixel moved since then (uv); when given, the track is carried
                along the motion (Lucas–Kanade flow)
            edge: Depth step within a 3×3 of depth samples that marks an edge, dropped (m); None takes `tolerance`
            max_speed: The fastest a body part moves in depth (m/s), the speed gate; 0 is none
            guard: The depth velocity's slot in the motion's FiniteGuard: a non-finite track 0 and counted; None no guard
        """
        if not self._allocated:
            return
        width, height = self._input_fbo.width, self._input_fbo.height
        step: tuple[float, float] = depth_step if depth_step[0] > 0.0 and depth_step[1] > 0.0 else (1.0 / width, 1.0 / height)
        alpha, _ = track_weights(delta_time, smooth, 1.0)

        FlowUtil.blit(self._raw, depth)
        self._input_fbo.swap()
        self._input_fbo.begin()
        self._step_filter_shader.use(self._raw, step, tolerance if edge is None else edge)
        self._input_fbo.end()

        # one pass reads the track and the catch-up of the frame before, and writes both
        self._estimate.swap()
        self._catch_up.swap()
        self._estimate.begin()
        self._velocity_shader.use(self._input_fbo, self._estimate.back, self._catch_up.back, self._catch_up,
                                  float(alpha), smooth, catch_up_smooth, float(alpha), motion_floor, max(delta_time, 1e-6), view_span,
                                  step, window, tolerance, dead_zone, displacement, max_speed, guard)
        self._estimate.end()
        glMemoryBarrier(int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT))

        self._output_fbo.begin()
        self._threshold_shader.use(self._estimate, threshold / max(width, 1))     # px as view widths
        self._output_fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def filter_depth_steps(depth: np.ndarray, step: tuple[float, float], tolerance: float) -> np.ndarray:
    """The depth (H, W, m) without its edges: a sample whose 3×3 depth samples, `step` pixels apart
    (x, y), span a step beyond `tolerance` (m) from the nearest to the furthest has no depth (0).
    At a silhouette that is the band where stereo depth jitters between the two surfaces. Mirrors
    depth_step_filter.frag."""
    h, w = depth.shape
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    near: np.ndarray = np.full((h, w), np.inf)
    far: np.ndarray = np.zeros((h, w))
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            x = np.clip(np.floor(columns + 0.5 + i * step[0]).astype(int), 0, w - 1)
            y = np.clip(np.floor(rows + 0.5 + j * step[1]).astype(int), 0, h - 1)
            sample: np.ndarray = depth[y, x]
            valid: np.ndarray = sample > 0.0
            near = np.where(valid, np.minimum(near, sample), near)
            far = np.where(valid, np.maximum(far, sample), far)
    edge: np.ndarray = np.isfinite(near) & (far - near > tolerance)
    return np.where(edge, 0.0, depth)


def depth_threshold(value: np.ndarray, threshold: float) -> np.ndarray:
    """A signed field with `threshold` taken off its size, 0 below it: the noise floor goes, and what is above stays continuous."""
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)
