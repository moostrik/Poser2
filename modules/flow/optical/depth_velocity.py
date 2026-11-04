"""Depth Velocity.

What the image cannot measure: the player's motion toward the camera, from the depth image. One
depth sample is mostly noise from frame to frame, and the change between two frames doubles it, so
z is tracked instead of differenced: each pixel holds a depth and its rate, carried along the motion
from frame to frame, and each new depth only corrects them (an alpha–beta filter). Over `smooth` (ms)
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
passing over the pixel, the track starts over with no rate. With `smooth` 0 and a full window it is
the plain change between two frames. A change that ramps past `tolerance` over a few frames, as the
background a moving limb uncovers, never misses the prediction by that much in one; with
`restart_frames` above 1, the track also starts over where the depth at the pixel, in place, is
further than `tolerance` from its depth 2 to `restart_frames` frames before.

A surface's depth wanders by a few mm, and its steps are a mm; the track reads both as slow motion.
With `dead_zone` (m), each pixel holds an anchor: while the track stays within the dead zone of it,
z is 0; past it, z reads at its full speed and the anchor is dragged along behind, so a real motion
loses only its first `dead_zone`.

Averaged over `smooth`, the track lags a fast motion toward the camera; past `tolerance` it would start
over and over. With `catch_up_smooth`, it catches up, as a tracker does a manoeuvring target: Emas of
its miss of the prediction, signed and its size, tell a lag (it misses one way) from jitter (it
misses both ways, and cancels out), and where the image moves too (`color_motion`: a still edge's
jitter has none, a hand pushing forward moves its edges) the time it follows over blends toward
`catch_up_smooth`. The carry then takes each texel's depth a frame on, where it lags to.

z comes in view widths per frame, over the metres one view width spans at the depth (`view_span`
per metre), and `threshold` comes off it. The Lucas–Kanade flow owns one for its z, with its
`depth_track` settings, and turns it into view widths per second times its `z_gain`.

At a silhouette, stereo depth jitters between the two surfaces by more than the dead zone, from
frame to frame. So before the track, a depth sample whose 3×3 depth samples span a step beyond
`tolerance` has no depth: the same test of "another surface" as the window's. Along each silhouette
a band a depth sample wide reads no z; the player's mask cuts there anyway.

`filter_depth_steps`, `track_depth`, `shift_history` and `depth_threshold` below are the numpy references of the shaders.
"""
import math

import numpy as np
from OpenGL.GL import GL_R16F, GL_R32F, GL_RG32F, GL_RGBA32F

from modules.gl import Fbo, SwapFbo, Texture
from .. import FlowBase, FlowUtil
from .shaders import DepthHistory, DepthStepFilter, DepthThreshold, DepthVelocityShader

RESTART_FRAMES: int = 4     # the most frames back the restart compares with: the history's four channels


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
        2. Correct the track carried from the last one → the track SwapFbo (RGBA32F: z, disparity, rate, anchor),
           and in a second run of the same shader its catch-up → the catch-up SwapFbo (RG32F)
        3. Move the depth's history one frame on → the history SwapFbo (RGBA32F: 1 to 4 frames before, m)
        4. Take the noise floor off z → output_fbo (R16F)
    """

    def __init__(self) -> None:
        super().__init__()

        self._input_internal_format = GL_R32F
        self._output_internal_format = GL_R16F

        self._step_filter_shader: DepthStepFilter = DepthStepFilter()
        self._velocity_shader: DepthVelocityShader = DepthVelocityShader()
        self._history_shader: DepthHistory = DepthHistory()
        self._threshold_shader: DepthThreshold = DepthThreshold()

        self._raw: Fbo = Fbo()                   # the depth as it came, before its edges are taken off
        self._track: SwapFbo = SwapFbo()
        self._catch_up: SwapFbo = SwapFbo()      # the Emas of the miss of the prediction, signed and its size (RG32F, m)
        self._history: SwapFbo = SwapFbo()       # the depth 1 to 4 frames before, in place (RGBA32F, m)

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
        self._catch_up.allocate(width, height, GL_RG32F)
        self._catch_up.clear_all()
        self._history.allocate(width, height, GL_RGBA32F)
        self._history.clear_all()
        self._step_filter_shader.allocate()
        self._velocity_shader.allocate()
        self._history_shader.allocate()
        self._threshold_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._raw.deallocate()
        self._track.deallocate()
        self._catch_up.deallocate()
        self._history.deallocate()
        self._step_filter_shader.deallocate()
        self._velocity_shader.deallocate()
        self._history_shader.deallocate()
        self._threshold_shader.deallocate()

    def reset(self) -> None:
        super().reset()
        self._track.clear_all()
        self._catch_up.clear_all()
        self._history.clear_all()

    def update_depth(self, depth: Texture, delta_time: float, view_span: float, depth_step: tuple[float, float], window: int,
                     smooth: float, threshold: float, tolerance: float, dead_zone: float, catch_up_smooth: float, motion_floor: float,
                     displacement: Texture | None = None, edge: float | None = None, restart_frames: int = 1) -> None:
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
            restart_frames: Frames back a depth change past `tolerance` starts the track over, 1 to
                `RESTART_FRAMES`; 1 is the frame before only
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

        # both runs read the track and the catch-up of the frame before: first the track, then the catch-up
        self._track.swap()
        self._catch_up.swap()
        for target, catch_up_pass in ((self._track, False), (self._catch_up, True)):
            target.begin()
            self._velocity_shader.use(self._input_fbo.texture, self._track.back_texture, self._catch_up.back_texture, catch_up_pass,
                                      float(alpha), smooth, catch_up_smooth, float(alpha), motion_floor, max(delta_time, 1e-6), view_span,
                                      step, window, tolerance, dead_zone, self._history.texture,
                                      min(max(restart_frames, 1), RESTART_FRAMES), displacement)
            target.end()

        self._history.swap()
        self._history.begin()
        self._history_shader.use(self._input_fbo.texture, self._history.back_texture)
        self._history.end()

        self._output_fbo.begin()
        self._threshold_shader.use(self._track.texture, threshold / max(width, 1))     # px as view widths
        self._output_fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def _carried(state: np.ndarray, catch_up: np.ndarray, displacement: np.ndarray, measurement: np.ndarray, tolerance: float,
             ahead: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The track's disparity, rate and anchor (H, W) and its catch-up (H, W, 2) where each pixel was a
    displacement back: linear between the four texels around it, of those that hold a track of the
    same surface as the measurement, within `tolerance` (m) of it, as the shader reads them; each
    texel's depth taken `ahead` seconds on by its rate. None of them: no track."""
    h, w = state.shape[:2]
    carried_catch_up: np.ndarray = np.zeros((h, w, 2))
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
            texel_depth: np.ndarray = 1.0 / np.where(held, texel[..., 1] + texel[..., 2] * ahead, 1.0)
            weight: np.ndarray = wx * wy * (held & (measurement > 0.0) & (np.abs(texel_depth - measured_depth) <= tolerance))
            disparity += weight * texel[..., 1]
            rate += weight * texel[..., 2]
            anchor += weight * texel[..., 3]
            carried_catch_up += weight[..., None] * catch_up[ty, tx]
            total += weight
    found: np.ndarray = total > 1e-9
    safe: np.ndarray = np.where(found, total, 1.0)
    return (np.where(found, disparity / safe, 0.0), np.where(found, rate / safe, 0.0), np.where(found, anchor / safe, 0.0),
            np.where(found[..., None], carried_catch_up / safe[..., None], 0.0))


CATCH_UP_FLOOR: float = 0.005       # m: what the index is taken over besides the miss, with no dead zone


def catch_up_share(signed: np.ndarray, size: np.ndarray, dead_zone: float) -> np.ndarray:
    """How far a track is catching up a motion, 0 to 1, from the Emas of its miss: signed and its size
    (m). A motion the track lags misses the same way, so the signed mean is as large as the size; a
    depth jittering back and forth cancels out in it. The signed mean over the size and the dead zone
    twice (so a miss within it never counts) is the manoeuvre index of tracking; the share runs
    smoothly from 0 at an index of 0.3 to 1 at 0.7."""
    index: np.ndarray = np.abs(signed) / (size + 2.0 * max(dead_zone, CATCH_UP_FLOOR))
    t: np.ndarray = np.clip((index - 0.3) / 0.4, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def color_motion(displacement: np.ndarray, motion_floor: float) -> np.ndarray:
    """Where the image moves (H, W): the largest displacement (pixels per frame) of the 3×3 pixels
    around each is past `motion_floor`. A real motion toward the camera moves its edges in the image
    too; a depth jittering at a still edge does not."""
    h, w = displacement.shape[:2]
    speed: np.ndarray = np.hypot(displacement[..., 0], displacement[..., 1])
    rows, columns = np.mgrid[0:h, 0:w]
    largest: np.ndarray = np.zeros((h, w))
    for j in (-1, 0, 1):
        for i in (-1, 0, 1):
            largest = np.maximum(largest, speed[np.clip(rows + j, 0, h - 1), np.clip(columns + i, 0, w - 1)])
    return largest > motion_floor


def track_depth(depth: np.ndarray, state: np.ndarray, catch_up: np.ndarray, displacement: np.ndarray, delta_time: float, window: int,
                step: tuple[float, float], tolerance: float, smooth: float, view_span: float, dead_zone: float,
                catch_up_smooth: float, motion_floor: float, history: np.ndarray | None = None,
                restart_frames: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The depth track one frame on (H, W, 4): z in view widths per frame, disparity (1/m), its rate
    (1/m per second) and its anchor (m); and its catch-up (H, W, 2): the Emas of its miss of the
    prediction, signed and its size (m). As the shader writes them, in its two runs; `catch_up` is
    the one the frame before.

    `depth` is in metres (H, W), 0 where there is none; `state` is the track the frame before
    (zeros to start); `displacement` is how far each pixel moved (H, W, 2, pixels per frame, x along
    columns, y along rows); `step` is the pixels between two depth samples (x, y). The measurement is
    the mean disparity of the (2·window+1)² samples within `tolerance` of the pixel's depth; it
    corrects the track carried from a displacement back, of the texels there within `tolerance` of
    it, by the share of the window it holds. Further than `tolerance` from the prediction, or
    without a track of its surface there, the track starts over with no rate. Without depth at the pixel or without a span, it is empty.
    With `dead_zone` (m) above 0, z is 0 while the tracked depth stays within it of the anchor; past
    it, z reads at its full speed and the anchor follows at `dead_zone` behind. With `catch_up_smooth`
    (ms) above 0, a track that keeps missing one way, as it lags a fast motion, follows over a time
    between `smooth` and `catch_up_smooth` by its `catch_up_share`, where the image moves (its
    `color_motion` past `motion_floor`, px per frame); and the carry takes each texel's depth a frame
    on by its rate. 0 is neither. With `restart_frames` above 1, the track also starts over where
    `depth` is further than `tolerance` from `history` (H, W, 4: its depth 1 to 4 frames before, in
    place, 0 where none, as `shift_history` keeps it) 2 to `restart_frames` frames before.
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

    dt: float = max(delta_time, 1e-6)
    catching: bool = catch_up_smooth > 0.0
    carried, carried_rate, carried_anchor, carried_catch_up = _carried(state, catch_up, displacement, measurement, tolerance,
                                                                       dt if catching else 0.0)
    predicted: np.ndarray = np.where(carried > 0.0, carried + carried_rate * dt, 0.0)
    safe_measurement: np.ndarray = np.where(measured, measurement, 1.0)
    safe_predicted: np.ndarray = np.where(predicted > 0.0, predicted, 1.0)
    miss: np.ndarray = np.where(measured & (predicted > 0.0), 1.0 / safe_measurement - 1.0 / safe_predicted, 0.0)   # m
    follows: np.ndarray = measured & (predicted > 0.0) & (np.abs(miss) <= tolerance)
    if history is not None:
        for k in range(1, min(max(restart_frames, 1), RESTART_FRAMES)):
            before: np.ndarray = history[..., k]
            follows &= ~((before > 0.0) & (np.abs(depth - before) > tolerance))

    # catching up (a manoeuvre, in tracking terms): the Emas of the miss over `smooth`, and the time the
    # track follows over blended toward `catch_up_smooth` by how much the miss keeps one way, where the
    # image moves too; a still edge's jitter never catches up
    ema: float = float(track_weights(delta_time, smooth, 1.0)[0])
    signed: np.ndarray = np.where(follows, carried_catch_up[..., 0] + ema * (miss - carried_catch_up[..., 0]), 0.0)
    size: np.ndarray = np.where(follows, carried_catch_up[..., 1] + ema * (np.abs(miss) - carried_catch_up[..., 1]), 0.0)
    if catching:
        opened: np.ndarray = catch_up_share(signed, size, dead_zone) * color_motion(displacement, motion_floor)
        follow_time: np.ndarray = smooth + opened * (catch_up_smooth - smooth)
        full: np.ndarray = 1.0 - np.exp(-delta_time * 1000.0 / np.maximum(follow_time, 1e-6)) if smooth > 0.0 else np.ones_like(follow_time)
        alpha = full * share
        beta = alpha * alpha / (2.0 - alpha)
    else:
        alpha, beta = track_weights(delta_time, smooth, share)
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
    return (np.where(measured[..., None], np.stack((z, disparity, rate, anchor), axis=-1), 0.0),
            np.stack((signed, size), axis=-1))


def shift_history(depth: np.ndarray, history: np.ndarray) -> np.ndarray:
    """The depth's history (H, W, 4, m) one frame on: `depth` (H, W) first, then the history's first
    three. Mirrors depth_history.frag."""
    return np.concatenate((depth[..., None], history[..., :3]), axis=-1)


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
