"""Lucas–Kanade Flow: how far each pixel moved between two luminance frames, coarse to fine.

Where the gradient flow gives a motion strength per pixel (the brightness change along the gradient),
this gives how far the image moved: a window of gradients per pixel, solved for the shift that carries
the previous frame onto the current one, first on the coarsest mip level, then refined once per finer
level with the previous frame warped by the estimate so far. With `xy_median`, the 3×3 or 5×5 median of the
displacement after each level, so one level's outliers do not grow on the next.

The displacement (uv per frame) stays inside: `xy_threshold` (px per frame) comes off its length, as
below it is the solve's jitter, not motion. z comes from its own `DepthVelocity`: a depth track carried along
the displacement, so a surface sliding sideways over a slope reads no motion toward the camera,
corrected each frame by the mean over `z_window` depth samples and smoothed over `z_smooth`,
starting over where the depth changed past `z_tolerance` within `z_restart_frames` frames;
`z_threshold` comes off it. The depth change comes in view widths per frame like the displacement, so all three become
view widths per second the same way. Out comes one 3D velocity: x/y times `x_gain`/`y_gain`,
z times `z_gain`. With `use_mask`, the velocity is multiplied by the mask.

z lags its motion by the time it is smoothed over. With `xy_smooth`, x/y goes through the same
filter over `z_smooth` before its threshold, carried along the displacement as z is, so both lag
alike. A displacement is a step of position, so the filter is the same one: with a steady frame
interval and a full z window, x/y answers a change of speed exactly as z does. The tracks are
carried by the displacement as solved, before `xy_threshold`: a slide too slow for the output
still moves the tracks with it, or on a slope it would read as z.

The settings in flow pixels go with the flow's resolution: `xy_levels` (each level doubles the motion
it follows, about 2 × 2^(levels − 1) px per frame at full size, so halving the resolution needs one
level fewer for the same motion), `xy_window`, the median, `xy_threshold` and `z_threshold`. Those in
metres and milliseconds do not.

`lucas_kanade`, `median_displacement`, `threshold_displacement`, `filter_displacement` and
`displacement_velocity` below are the numpy references of the shaders.
"""

import numpy as np
from OpenGL.GL import GL_RG32F, GL_RGBA16F, GL_RGBA32F

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field, Widget
from .. import FlowUtil
from .depth_velocity import DepthVelocity, track_weights
from .luminance_frames import LuminanceFrames
from .patch import Patch
from .shaders import ComposeVelocity, DisplacementFilter, DisplacementMedian, DisplacementThreshold, LucasKanadeShader


class LucasKanadeFlowSettings(BaseSettings):
    """Configuration for the Lucas–Kanade flow."""
    xy_levels:      Field[int]   = Field(3, min=3, max=5, description="Pyramid levels of the x/y solve, each doubles the reach (more: faster motion, more spill)")
    xy_window:      Field[Patch] = Field(Patch.FIVE, min=Patch.THREE, max=Patch.NINE, description="x/y patch size (larger: cleaner, more spill)")
    xy_damping:     Field[float] = Field(0.001, min=0.0, max=0.1, step=0.001, description="Damps x/y where the patch has little texture (higher: less noise, smaller motion)")
    xy_same_surface: Field[bool] = Field(False, newline=True, width=48, description="Patch counts only the center's surface (on: less smear, wide fast parts slower)")
    xy_threshold:   Field[float] = Field(0.5, min=0.0, max=5.0, step=0.05, description="x/y below this is noise, in px/frame (higher: less noise, less slow motion)")
    xy_median:      Field[Patch] = Field(Patch.ONE, min=Patch.ONE, max=Patch.FIVE, visible=False, description="Median size per level, 1×1 = off (larger: cleaner, more GPU)")  # hidden: kept for hd_trio's 2D LK at a lower resolution; delete it if it doesn't earn its place there
    z_edge:         Field[float] = Field(0.1, min=0.05, max=0.3, step=0.005, newline=True, description="Edge depth step, m (lower: cleaner edges, fewer fingers)")
    z_window:       Field[Patch] = Field(Patch.THREE, min=Patch.ONE, max=Patch.NINE, description="Depth averaging size (larger: less noise, small parts blur)")
    z_tolerance:    Field[float] = Field(0.1, min=0.05, max=0.5, step=0.01, description="New-surface depth jump, m (higher: fuller pushes, fewer splits)")
    z_restart_frames: Field[int] = Field(1, min=1, max=4, description="Frames a jump past z_tolerance may take, 1 = one (more: less wake z, fast pushes cut)")
    z_smooth:       Field[float] = Field(50.0, min=0.0, max=200.0, step=1.0, newline=True, description="Smoothing time of z in ms, 0 = two-frame change (higher: less noise, more lag)")
    xy_smooth:      Field[bool]  = Field(False, width=48, description="x/y smoothed over z_smooth, so both lag alike (on: less noise, more lag)")
    z_catch_up:     Field[float] = Field(0.0, min=0.0, max=50.0, step=1.0, description="Shorter smoothing during a fast push, in ms, 0 = off (lower: quicker pushes)")
    z_dead_zone:    Field[float] = Field(0.0, min=0.0, max=0.05, step=0.001, newline=True, description="Depth drift ignored before z counts as motion, in m (higher: less drift, slower start)")
    z_threshold:    Field[float] = Field(0.5, min=0.0, max=5.0, step=0.05, description="z below this is noise, in px/frame (higher: less noise, less slow z)")
    use_mask:       Field[bool]  = Field(False, newline=True, width=48, description="Only the player moves the fluid (off: whole image)")
    x_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, widget=Widget.knob, description="1 = true speed, image widths per second")
    y_gain:         Field[float] = Field(1.0, min=-10.0, max=10.0, widget=Widget.knob, description="1 = true speed, image widths per second")
    z_gain:         Field[float] = Field(0.0, min=-10.0, max=10.0, widget=Widget.knob, description="1 = true speed, on the x/y scale")


class LucasKanadeFlow:
    """The 3D velocity (RGBA16F, xyz) between the newest two frames of its input."""

    def __init__(self, settings: LucasKanadeFlowSettings) -> None:
        self.settings: LucasKanadeFlowSettings = settings
        self._depth_velocity: DepthVelocity = DepthVelocity()
        self._mask: Texture | None = None

        self._frames: LuminanceFrames = LuminanceFrames()
        self._solve: SwapFbo = SwapFbo()            # the displacement as solved, level by level
        self._displacement: Fbo = Fbo()             # with the noise floor taken off
        self._filtered: SwapFbo = SwapFbo()         # with xy_smooth: the filter, its displacement and position error
        self._followed: Fbo = Fbo()                 # the filtered displacement with the noise floor taken off
        self._following: bool = False
        self._velocity: Fbo = Fbo()
        self._shader: LucasKanadeShader = LucasKanadeShader()
        self._threshold_shader: DisplacementThreshold = DisplacementThreshold()
        self._median_shader: DisplacementMedian = DisplacementMedian()
        self._filter_shader: DisplacementFilter = DisplacementFilter()
        self._compose_shader: ComposeVelocity = ComposeVelocity()

    @property
    def luminance(self) -> Texture:
        """The newest luminance frame."""
        return self._frames.luminance

    @property
    def velocity(self) -> Texture:
        """x/y in image widths per second and z, each times its strength (RGBA16F)."""
        return self._velocity

    @property
    def depth_velocity(self) -> DepthVelocity:
        """Its z, before `z_gain` and the mask."""
        return self._depth_velocity

    def allocate(self, width: int, height: int) -> None:
        self._depth_velocity.allocate(width, height)
        self._frames.allocate(width, height)
        self._solve.allocate(width, height, GL_RG32F)
        self._displacement.allocate(width, height, GL_RG32F)
        self._filtered.allocate(width, height, GL_RGBA32F)
        self._followed.allocate(width, height, GL_RG32F)
        self._velocity.allocate(width, height, GL_RGBA16F)
        self._shader.allocate()
        self._threshold_shader.allocate()
        self._median_shader.allocate()
        self._filter_shader.allocate()
        self._compose_shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._depth_velocity.deallocate()
        self._frames.deallocate()
        self._solve.deallocate()
        self._displacement.deallocate()
        self._filtered.deallocate()
        self._followed.deallocate()
        self._velocity.deallocate()
        self._shader.deallocate()
        self._threshold_shader.deallocate()
        self._median_shader.deallocate()
        self._filter_shader.deallocate()
        self._compose_shader.deallocate()

    def reset(self) -> None:
        self._depth_velocity.reset()
        self._frames.reset()
        self._solve.clear_all()
        self._displacement.clear()
        self._filtered.clear_all()
        self._followed.clear()
        self._velocity.clear()

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._frames.set_input(texture)

    def set_mask(self, texture: Texture | None) -> None:
        """The player's mask (red, 0 to 1) the velocity is multiplied by with `use_mask`, or None for no mask."""
        self._mask = texture

    def _pyramid(self, solve: SwapFbo, previous: Texture, current: Texture, depth: Texture | None) -> None:
        """The displacement carrying `previous` onto `current`, coarse to fine, into `solve`: one solve
        per level, and with `xy_median` the median after each level."""
        s: LucasKanadeFlowSettings = self.settings
        solve.clear()                                       # the coarsest level starts from no motion
        for level in reversed(range(s.xy_levels)):
            solve.swap()
            solve.begin()
            self._shader.use(previous, current, solve.back_texture, level, s.xy_window, s.xy_damping,
                             depth if s.xy_same_surface else None, s.z_tolerance)
            solve.end()
            if s.xy_median:
                spacing: float = float(2 ** level)          # a level's texel apart
                solve.swap()
                solve.begin()
                self._median_shader.use(solve.back_texture, (spacing / solve.width, spacing / solve.height), int(s.xy_median))
                solve.end()

    def update(self, depth: Texture | None, delta_time: float, view_span: float, depth_step: tuple[float, float]) -> None:
        """Measure between the newest two frames.

        Args:
            depth: Metres in red; None without depth (z is 0)
            delta_time: Seconds since the previous frame
            view_span: Metres one view width spans per metre of depth, for z
            depth_step: uv between two depth samples in the flow's view, for z
        """
        if not self._frames.ready:
            FlowUtil.zero(self._velocity)
            return
        s: LucasKanadeFlowSettings = self.settings

        self._pyramid(self._solve, self._frames.previous_luminance, self._frames.luminance, depth)
        solved: Texture = self._solve.texture

        self._displacement.begin()
        self._threshold_shader.use(solved, s.xy_threshold)
        self._displacement.end()

        if depth is not None:
            self._depth_velocity.update_depth(depth, delta_time, view_span, depth_step, s.z_window, s.z_smooth,
                                              s.z_threshold, s.z_tolerance, s.z_dead_zone, s.z_catch_up, s.xy_threshold,
                                              solved, s.z_edge, s.z_restart_frames)

        xy: Texture = self._displacement
        if s.xy_smooth:
            if not self._following:
                self._filtered.clear_all()                  # from no motion, not from when it was last on
                self._following = True
            alpha, _ = track_weights(delta_time, s.z_smooth, 1.0)
            self._filtered.swap()
            self._filtered.begin()
            self._filter_shader.use(solved, self._filtered.back_texture, solved, float(alpha))
            self._filtered.end()
            self._followed.begin()
            self._threshold_shader.use(self._filtered.texture, s.xy_threshold)
            self._followed.end()
            xy = self._followed
        else:
            self._following = False

        # uv per frame as image widths per second: v is uv of the height, so a square motion stays square;
        # z is in view widths per frame already
        per_second: float = 1.0 / max(delta_time, 1e-6)
        aspect: float = self._displacement.height / max(self._displacement.width, 1)
        mask: Texture | None = self._mask if s.use_mask and self._mask is not None and self._mask.allocated else None
        self._velocity.begin()
        self._compose_shader.use(xy, self._depth_velocity.depth_velocity if depth is not None else None,
                                 (s.x_gain * per_second, s.y_gain * per_second * aspect, s.z_gain * per_second), mask)
        self._velocity.end()


# ── numpy reference ─────────────────────────────────────────────────────

def _pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """The image and its mip levels, each the 2x2 mean of the one before, as GL builds them."""
    pyramid: list[np.ndarray] = [image]
    for _ in range(1, levels):
        top: np.ndarray = pyramid[-1]
        h, w = top.shape[0] // 2, top.shape[1] // 2
        pyramid.append(top[:h * 2, :w * 2].reshape(h, 2, w, 2).mean(axis=(1, 3)))
    return pyramid


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`image` sampled at pixel coordinates (centers at integers), linear between them, clamped at the edges."""
    h, w = image.shape
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top: np.ndarray = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bottom: np.ndarray = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _solve_level(p_image: np.ndarray, c_image: np.ndarray, d: np.ndarray, u: np.ndarray, v: np.ndarray, window: int,
                 damping: float, depth: np.ndarray | None = None, tolerance: float = 0.0) -> np.ndarray:
    """One solve on one level: the displacement so far (uv) refined by the window's shift, with `depth`
    only its samples of the centre's surface. Mirrors the shader."""
    h, w = u.shape
    lh, lw = c_image.shape
    sxx = sxy = syy = sxt = syt = np.zeros((h, w))
    total, count = np.zeros((h, w)), 0
    if depth is not None:
        dh, dw = depth.shape
        center = depth[np.clip(np.floor(v * dh).astype(int), 0, dh - 1), np.clip(np.floor(u * dw).astype(int), 0, dw - 1)]
    for j in range(-window, window + 1):
        for i in range(-window, window + 1):
            px, py = u * lw - 0.5 + i, v * lh - 0.5 + j
            weight: np.ndarray = np.ones((h, w))
            if depth is not None:
                su, sv = u + i / lw, v + j / lh
                sample = depth[np.clip(np.floor(sv * dh).astype(int), 0, dh - 1), np.clip(np.floor(su * dw).astype(int), 0, dw - 1)]
                same = (sample > 0.0) & (np.abs(sample - center) <= tolerance)
                weight = np.where(center > 0.0, weight * same, weight)
            total, count = total + weight, count + 1
            qx, qy = px - d[..., 0] * lw, py - d[..., 1] * lh
            gx = 0.25 * (_bilinear(c_image, px + 1, py) - _bilinear(c_image, px - 1, py)
                         + _bilinear(p_image, qx + 1, qy) - _bilinear(p_image, qx - 1, qy))
            gy = 0.25 * (_bilinear(c_image, px, py + 1) - _bilinear(c_image, px, py - 1)
                         + _bilinear(p_image, qx, qy + 1) - _bilinear(p_image, qx, qy - 1))
            gt = _bilinear(c_image, px, py) - _bilinear(p_image, qx, qy)
            sxx, sxy, syy = sxx + weight * gx * gx, sxy + weight * gx * gy, syy + weight * gy * gy
            sxt, syt = sxt + weight * gx * gt, syt + weight * gy * gt
    kept_damping: np.ndarray = damping * total / count
    sxx, syy = sxx + kept_damping, syy + kept_damping
    det: np.ndarray = sxx * syy - sxy * sxy
    safe: np.ndarray = np.where(np.abs(det) > 1e-12, det, 1.0)
    dx = np.where(np.abs(det) > 1e-12, -(syy * sxt - sxy * syt) / safe, 0.0)
    dy = np.where(np.abs(det) > 1e-12, -(sxx * syt - sxy * sxt) / safe, 0.0)
    return d + np.stack((dx / lw, dy / lh), axis=-1)


def median_displacement(displacement: np.ndarray, spacing: float, radius: int = 2) -> np.ndarray:
    """The displacement (H, W, 2), each component the median of its (2·radius+1)² samples `spacing`
    pixels apart, nearest, clamped at the edges. Mirrors displacement_median.frag."""
    h, w = displacement.shape[:2]
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    samples: list[np.ndarray] = []
    for j in range(-radius, radius + 1):
        for i in range(-radius, radius + 1):
            y = np.clip(np.floor(rows + 0.5 + j * spacing).astype(int), 0, h - 1)
            x = np.clip(np.floor(columns + 0.5 + i * spacing).astype(int), 0, w - 1)
            samples.append(displacement[y, x])
    return np.median(np.stack(samples), axis=0)


def lucas_kanade(previous: np.ndarray, current: np.ndarray, levels: int = 3, window: int = 2, damping: float = 0.001,
                 median: int = 0, depth: np.ndarray | None = None, tolerance: float = 0.0) -> np.ndarray:
    """The displacement (H, W, 2) in pixels per frame, x along columns and y along rows, that carries
    `previous` onto `current`: coarse to fine over `levels` mip levels, one solve per level over a
    (2·window+1)² window, the previous frame warped by the estimate so far; with `median`, after each
    level the median of the displacement over a (2·median+1)² patch, a level's texel apart. With `depth`
    (H, W, m), only the window's samples within `tolerance` of the centre's depth count. Mirrors the shaders."""
    h, w = current.shape
    previous_levels: list[np.ndarray] = _pyramid(previous.astype(np.float64), levels)
    current_levels: list[np.ndarray] = _pyramid(current.astype(np.float64), levels)
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = (columns + 0.5) / w, (rows + 0.5) / h            # each output pixel's uv
    d: np.ndarray = np.zeros((h, w, 2))                     # the displacement in uv
    for level in reversed(range(levels)):
        d = _solve_level(previous_levels[level], current_levels[level], d, u, v, window, damping, depth, tolerance)
        if median:
            d = median_displacement(d, float(2 ** level), median)
    return d * np.array([w, h])


def threshold_displacement(displacement: np.ndarray, threshold: float) -> np.ndarray:
    """A displacement (..., 2) in pixels with `threshold` pixels taken off its length, 0 below it. Mirrors displacement_threshold.frag."""
    length: np.ndarray = np.linalg.norm(displacement, axis=-1, keepdims=True)
    kept: np.ndarray = np.maximum(length - threshold, 0.0)
    return np.divide(displacement * kept, length, out=np.zeros_like(displacement), where=length > 1e-12)


def filter_displacement(displacement: np.ndarray, state: np.ndarray, carry: np.ndarray, alpha: float) -> np.ndarray:
    """The filter x/y goes through with xy_smooth, one frame on (H, W, 4): the filtered
    displacement (x, y, pixels per frame) and the position error (x, y, pixels). Mirrors
    displacement_filter.frag.

    The depth track's alpha–beta filter, on positions that step by `displacement` (H, W, 2, pixels
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
