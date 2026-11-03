"""Depth Velocity.

What the image cannot measure: the player's motion toward the camera, from the depth image. One
depth sample is mostly noise from frame to frame, and the change between two frames doubles it, so
z is tracked instead of differenced: each pixel holds a depth and its rate, carried along the motion
from frame to frame, and each new depth only corrects them (an alpha–beta filter). Over `time` (ms)
the corrections average out, so z is the rate of many frames rather than the step of two.

The track is held in disparity (1 / depth), where stereo depth noise is about constant. `tolerance`
is in metres, as it tells surfaces apart: a hand 10 cm before the chest is 10 cm at any distance;
the depth noise grows with distance, so far away more of it reads as another surface. Each frame,
the measurement is the mean disparity over a window of depth samples, of those within `tolerance`
of the pixel's depth (not the background beside the player); it corrects the track by the share of the window it holds, so a
thin window at a silhouette corrects little and the track coasts on. The track comes from where the
pixel was a displacement back (in place without one), so a surface sliding sideways keeps its own
depth, and only from the texels there of the same surface, within `tolerance` of the measurement:
where the displacement is unsure, at an edge, a track is never carried over from the other side. Where the measurement is further than `tolerance` from the track's prediction, as a limb
passing over the pixel, the track starts over with no rate. With `time` 0 and a full window it is
the plain change between two frames.

A surface's depth wanders by a few mm, and its steps are a mm; the track reads both as slow motion.
With `dead_zone` (m), each pixel holds an anchor: while the track stays within the dead zone of it,
z is 0; past it, z reads at its full speed and the anchor is dragged along behind, so a real motion
loses only its first `dead_zone`.

z comes in view widths per frame, over the metres one view width spans at the depth (`view_span`
per metre), and `threshold` comes off it. Each optical flow owns one for its z, with its own
`z_*` settings, and turns it into view widths per second times its own `strength_z`.

At a silhouette, stereo depth jitters between the two surfaces by more than the dead zone, from
frame to frame. So before the track, a depth sample whose 3×3 depth samples span a step beyond
`tolerance` has no depth: the same test of "another surface" as the window's. Along each silhouette
a band a depth sample wide reads no z; the player's mask cuts there anyway.

`filter_depth_steps`, `track_depth` and `depth_threshold` below are the numpy references of the shaders.
"""
import math

import numpy as np
from OpenGL.GL import GL_R16F, GL_R32F, GL_RGBA32F

from modules.gl import Fbo, SwapFbo, Texture
from .. import FlowBase, FlowUtil
from .shaders import DepthStepFilter, DepthThreshold, DepthVelocityShader


def track_weights(delta_time: float, time: float, share: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
    """The alpha–beta filter's weights: alpha on the depth, beta on the rate, for a measurement that
    holds `share` of its window. Alpha decays over `time` (ms), 1 at 0; beta is alpha² / (2 − alpha),
    critically damped, so the rate settles without overshoot."""
    alpha: float = 1.0 - math.exp(-delta_time * 1000.0 / time) if time > 0.0 else 1.0
    weighted = alpha * share
    return weighted, weighted * weighted / (2.0 - weighted)


class DepthVelocity(FlowBase):
    """Signed motion toward the camera in view widths per frame (R16F), positive toward it.

    Pipeline, on each new flow frame:
        1. Take the depth → the raw Fbo (R32F, metres, at the flow's size, nearest), and without
           its edges → the input SwapFbo
        2. Correct the track carried from the last one → the track SwapFbo (RGBA32F: z, disparity, rate, anchor)
        3. Take the noise floor off z → output_fbo (R16F)
    """

    def __init__(self) -> None:
        super().__init__()

        self._input_internal_format = GL_R32F
        self._output_internal_format = GL_R16F

        self._step_filter_shader: DepthStepFilter = DepthStepFilter()
        self._velocity_shader: DepthVelocityShader = DepthVelocityShader()
        self._threshold_shader: DepthThreshold = DepthThreshold()

        self._raw: Fbo = Fbo()                   # the depth as it came, before its edges are taken off
        self._track: SwapFbo = SwapFbo()

    @property
    def depth_velocity(self) -> Texture:
        """Motion toward the camera in view widths per frame (R16F)."""
        return self._output

    @property
    def difference(self) -> Texture:
        """The track (RGBA32F): z before the threshold in red (view widths per frame), its disparity
        (1/m) in green, the disparity's rate (1/m per second) in blue and the dead zone's anchor (m) in alpha."""
        return self._track.texture

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        super().allocate(width, height, output_width, output_height)
        self._raw.allocate(width, height, GL_R32F)
        self._track.allocate(width, height, GL_RGBA32F)
        self._track.clear_all()
        self._step_filter_shader.allocate()
        self._velocity_shader.allocate()
        self._threshold_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._raw.deallocate()
        self._track.deallocate()
        self._step_filter_shader.deallocate()
        self._velocity_shader.deallocate()
        self._threshold_shader.deallocate()

    def reset(self) -> None:
        super().reset()
        self._track.clear_all()

    def update_depth(self, depth: Texture, delta_time: float, view_span: float, depth_step: tuple[float, float], window: int,
                     time: float, threshold: float, tolerance: float, dead_zone: float, displacement: Texture | None = None) -> None:
        """Advance on a new flow frame: correct the track by the new depth, then take the noise floor off.

        Args:
            depth: Metres in red, 0 where there is none
            delta_time: Seconds since the last depth
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            depth_step: uv between two depth samples in the flow's view; 0 takes one flow texel
            window: Radius in depth samples
            time: Time over which the track averages (ms); 0 is the change of two frames
            threshold: Depth motion below this is noise, taken off the rest (px per frame)
            tolerance: Depth further than this is another surface: not averaged in, the track starts over (m)
            dead_zone: Depth the track wanders within its anchor without z (m); 0 is none
            displacement: How far each pixel moved since then (uv); when given, the track is carried
                along the motion (Lucas–Kanade flow)
        """
        if not self._allocated:
            return
        width, height = self._input_fbo.width, self._input_fbo.height
        step: tuple[float, float] = depth_step if depth_step[0] > 0.0 and depth_step[1] > 0.0 else (1.0 / width, 1.0 / height)
        alpha, _ = track_weights(delta_time, time, 1.0)

        FlowUtil.blit(self._raw, depth)
        self._input_fbo.swap()
        self._input_fbo.begin()
        self._step_filter_shader.use(self._raw, step, tolerance)
        self._input_fbo.end()

        self._track.swap()
        self._track.begin()
        self._velocity_shader.use(self._input_fbo.texture, self._track.back_texture, float(alpha), max(delta_time, 1e-6),
                                  view_span, step, window, tolerance, dead_zone, displacement)
        self._track.end()

        self._output_fbo.begin()
        self._threshold_shader.use(self._track.texture, threshold / max(width, 1))     # px as view widths
        self._output_fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def _carried(state: np.ndarray, displacement: np.ndarray, measurement: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The track's disparity, rate and anchor (H, W) where each pixel was a displacement back: linear
    between the four texels around it, of those that hold a track of the same surface as the
    measurement, within `tolerance` (m) of it, as the shader reads them. None of them: no track."""
    h, w = state.shape[:2]
    measured_depth: np.ndarray = 1.0 / np.where(measurement > 0.0, measurement, np.inf)
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = columns - displacement[..., 0], rows - displacement[..., 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    disparity: np.ndarray = np.zeros((h, w))
    rate: np.ndarray = np.zeros((h, w))
    anchor: np.ndarray = np.zeros((h, w))
    total: np.ndarray = np.zeros((h, w))
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            tx, ty = np.clip(x0 + dx, 0, w - 1), np.clip(y0 + dy, 0, h - 1)
            texel: np.ndarray = state[ty, tx]
            held: np.ndarray = texel[..., 1] > 0.0
            texel_depth: np.ndarray = 1.0 / np.where(held, texel[..., 1], 1.0)
            weight: np.ndarray = wx * wy * (held & (measurement > 0.0) & (np.abs(texel_depth - measured_depth) <= tolerance))
            disparity += weight * texel[..., 1]
            rate += weight * texel[..., 2]
            anchor += weight * texel[..., 3]
            total += weight
    found: np.ndarray = total > 1e-9
    safe: np.ndarray = np.where(found, total, 1.0)
    return np.where(found, disparity / safe, 0.0), np.where(found, rate / safe, 0.0), np.where(found, anchor / safe, 0.0)


def track_depth(depth: np.ndarray, state: np.ndarray, displacement: np.ndarray, delta_time: float, window: int,
                step: tuple[float, float], tolerance: float, time: float, view_span: float, dead_zone: float) -> np.ndarray:
    """The depth track one frame on (H, W, 4): z in view widths per frame, disparity (1/m), its rate
    (1/m per second) and its anchor (m), as the shader writes them.

    `depth` is in metres (H, W), 0 where there is none; `state` is the track the frame before
    (zeros to start); `displacement` is how far each pixel moved (H, W, 2, pixels per frame, x along
    columns, y along rows); `step` is the pixels between two depth samples (x, y). The measurement is
    the mean disparity of the (2·window+1)² samples within `tolerance` of the pixel's depth; it
    corrects the track carried from a displacement back, of the texels there within `tolerance` of
    it, by the share of the window it holds. Further than `tolerance` from the prediction, or
    without a track of its surface there, the track starts over with no rate. Without depth at the pixel or without a span, it is empty.
    With `dead_zone` (m) above 0, z is 0 while the tracked depth stays within it of the anchor; past
    it, z reads at its full speed and the anchor follows at `dead_zone` behind.
    """
    h, w = depth.shape
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    total: np.ndarray = np.zeros((h, w))
    kept: np.ndarray = np.zeros((h, w))
    for j in range(-window, window + 1):
        for i in range(-window, window + 1):
            x, y = columns + 0.5 + i * step[0], rows + 0.5 + j * step[1]
            sample: np.ndarray = depth[np.clip(np.floor(y).astype(int), 0, h - 1), np.clip(np.floor(x).astype(int), 0, w - 1)]
            keep: np.ndarray = (sample > 0.0) & (np.abs(sample - depth) <= tolerance)
            total += np.where(keep, 1.0 / np.where(keep, sample, 1.0), 0.0)
            kept += keep
    measured: np.ndarray = (depth > 0.0) & (view_span > 0.0) & (kept > 0)
    measurement: np.ndarray = np.where(measured, total / np.maximum(kept, 1), 0.0)
    share: np.ndarray = kept / float((2 * window + 1) ** 2)

    carried, carried_rate, carried_anchor = _carried(state, displacement, measurement, tolerance)
    dt: float = max(delta_time, 1e-6)
    predicted: np.ndarray = np.where(carried > 0.0, carried + carried_rate * dt, 0.0)
    safe_measurement: np.ndarray = np.where(measured, measurement, 1.0)
    safe_predicted: np.ndarray = np.where(predicted > 0.0, predicted, 1.0)
    follows: np.ndarray = measured & (predicted > 0.0) & (np.abs(1.0 / safe_measurement - 1.0 / safe_predicted) <= tolerance)

    alpha, beta = track_weights(delta_time, time, share)
    residual: np.ndarray = measurement - predicted
    disparity: np.ndarray = np.where(follows, predicted + alpha * residual, measurement)
    rate: np.ndarray = np.where(follows, carried_rate + beta * residual / dt, 0.0)
    z: np.ndarray = np.where(follows, rate * dt / (np.where(follows, disparity, 1.0) * max(view_span, 1e-9)), 0.0)

    # the dead zone: within `dead_zone` of its anchor the track reads no z; past it, z at its full
    # speed, and the anchor is dragged along (a backlash). A track that starts over is anchored anew
    tracked: np.ndarray = 1.0 / np.where(disparity > 0.0, disparity, 1.0)
    anchor: np.ndarray = np.where(follows & (carried_anchor > 0.0), carried_anchor, tracked)
    if dead_zone > 0.0:
        offset: np.ndarray = tracked - anchor
        outside: np.ndarray = np.abs(offset) > dead_zone
        anchor = np.where(outside, tracked - np.sign(offset) * dead_zone, anchor)
        z = np.where(outside, z, 0.0)
    else:
        anchor = tracked
    return np.where(measured[..., None], np.stack((z, disparity, rate, anchor), axis=-1), 0.0)


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
