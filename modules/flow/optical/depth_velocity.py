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
depth. Where the measurement is further than `tolerance` from the track's prediction, as a limb
passing over the pixel, the track starts over with no rate. With `time` 0 and a full window it is
the plain change between two frames.

z comes in view widths per frame, over the metres one view width spans at the depth (`view_span`
per metre), and `threshold` comes off it. Each optical flow owns one for its z, with its own
`depth_*` settings, and turns it into view widths per second times its own `strength_z`.

`track_depth` and `depth_threshold` below are the numpy references of the shaders.
"""
import math

import numpy as np
from OpenGL.GL import GL_R16F, GL_R32F, GL_RGBA32F

from modules.gl import SwapFbo, Texture
from .. import FlowBase, FlowUtil
from .shaders import DepthThreshold, DepthVelocityShader


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
        1. Take the depth → the input SwapFbo (R32F, metres, at the flow's size, nearest)
        2. Correct the track carried from the last one → the track SwapFbo (RGBA32F: z, disparity, rate)
        3. Take the noise floor off z → output_fbo (R16F)
    """

    def __init__(self) -> None:
        super().__init__()

        self._input_internal_format = GL_R32F
        self._output_internal_format = GL_R16F

        self._velocity_shader: DepthVelocityShader = DepthVelocityShader()
        self._threshold_shader: DepthThreshold = DepthThreshold()

        self._track: SwapFbo = SwapFbo()

    @property
    def depth_velocity(self) -> Texture:
        """Motion toward the camera in view widths per frame (R16F)."""
        return self._output

    @property
    def difference(self) -> Texture:
        """The track (RGBA32F): z before the threshold in red (view widths per frame), its disparity
        (1/m) in green and the disparity's rate (1/m per second) in blue."""
        return self._track.texture

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        super().allocate(width, height, output_width, output_height)
        self._track.allocate(width, height, GL_RGBA32F)
        self._track.clear_all()
        self._velocity_shader.allocate()
        self._threshold_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._track.deallocate()
        self._velocity_shader.deallocate()
        self._threshold_shader.deallocate()

    def reset(self) -> None:
        super().reset()
        self._track.clear_all()

    def update_depth(self, depth: Texture, delta_time: float, view_span: float, depth_step: tuple[float, float], window: int,
                     time: float, threshold: float, tolerance: float, displacement: Texture | None = None) -> None:
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
            displacement: How far each pixel moved since then (uv); when given, the track is carried
                along the motion (Lucas–Kanade flow)
        """
        if not self._allocated:
            return
        width, height = self._input_fbo.width, self._input_fbo.height
        step: tuple[float, float] = depth_step if depth_step[0] > 0.0 and depth_step[1] > 0.0 else (1.0 / width, 1.0 / height)
        alpha, _ = track_weights(delta_time, time, 1.0)

        self._input_fbo.swap()
        FlowUtil.blit(self._input_fbo, depth)

        self._track.swap()
        self._track.begin()
        self._velocity_shader.use(self._input_fbo.texture, self._track.back_texture, float(alpha), max(delta_time, 1e-6),
                                  view_span, step, window, tolerance, displacement)
        self._track.end()

        self._output_fbo.begin()
        self._threshold_shader.use(self._track.texture, threshold / max(width, 1))     # px as view widths
        self._output_fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def _carried(state: np.ndarray, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The track's disparity and rate (H, W) where each pixel was a displacement back: linear between
    the four texels around it, of those that hold a track, as the shader reads them."""
    h, w = state.shape[:2]
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    x, y = columns - displacement[..., 0], rows - displacement[..., 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    disparity: np.ndarray = np.zeros((h, w))
    rate: np.ndarray = np.zeros((h, w))
    total: np.ndarray = np.zeros((h, w))
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            tx, ty = np.clip(x0 + dx, 0, w - 1), np.clip(y0 + dy, 0, h - 1)
            texel: np.ndarray = state[ty, tx]
            weight: np.ndarray = wx * wy * (texel[..., 1] > 0.0)
            disparity += weight * texel[..., 1]
            rate += weight * texel[..., 2]
            total += weight
    held: np.ndarray = total > 1e-9
    return np.where(held, disparity / np.where(held, total, 1.0), 0.0), np.where(held, rate / np.where(held, total, 1.0), 0.0)


def track_depth(depth: np.ndarray, state: np.ndarray, displacement: np.ndarray, delta_time: float, window: int,
                step: tuple[float, float], tolerance: float, time: float, view_span: float) -> np.ndarray:
    """The depth track one frame on (H, W, 3): z in view widths per frame, disparity (1/m), its rate
    (1/m per second), as the shader writes them.

    `depth` is in metres (H, W), 0 where there is none; `state` is the track the frame before
    (zeros to start); `displacement` is how far each pixel moved (H, W, 2, pixels per frame, x along
    columns, y along rows); `step` is the pixels between two depth samples (x, y). The measurement is
    the mean disparity of the (2·window+1)² samples within `tolerance` of the pixel's depth; it
    corrects the track carried from a displacement back by the share of the window it holds. Further
    than `tolerance` from the prediction, or without a track there, the track starts over with no
    rate. Without depth at the pixel or without a span, it is empty.
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

    carried, carried_rate = _carried(state, displacement)
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
    return np.where(measured[..., None], np.stack((z, disparity, rate), axis=-1), 0.0)


def depth_threshold(value: np.ndarray, threshold: float) -> np.ndarray:
    """A signed field with `threshold` taken off its size, 0 below it: the noise floor goes, and what is above stays continuous."""
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)
