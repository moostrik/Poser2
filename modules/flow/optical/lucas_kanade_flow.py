"""Lucas–Kanade Flow: how far each pixel moved between two luminance frames, coarse to fine.

Where the gradient flow gives a motion strength per pixel (the brightness change along the gradient),
this gives how far the image moved: a window of gradients per pixel, solved for the shift that carries
the previous frame onto the current one, first on the coarsest mip level, then refined once per finer
level with the previous frame warped by the estimate so far.

The displacement (uv per frame) stays inside: `xy_threshold` (px per frame) comes off its length, as
below it is the solve's jitter, not motion. z comes from its own `DepthVelocity`: a depth track carried along
the displacement, so a surface sliding sideways over a slope reads no motion toward the camera,
corrected each frame by the mean over `depth_window` depth samples and averaged over `depth_time`;
`depth_threshold` comes off it. The depth change comes in view widths per frame like the displacement, so all three become
view widths per second the same way. Out comes one 3D velocity: x/y times `strength_x`/`strength_y`,
z times `strength_z`. With `use_mask`, the velocity is multiplied by the mask.

`lucas_kanade`, `threshold_displacement` and `displacement_velocity` below are the numpy references of the shaders.
"""

import numpy as np
from OpenGL.GL import GL_RG32F, GL_RGBA16F

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field
from .. import FlowUtil
from .depth_velocity import DepthVelocity
from .luminance_frames import LuminanceFrames
from .shaders import ComposeVelocity, DisplacementThreshold, LucasKanadeShader


class LucasKanadeFlowSettings(BaseSettings):
    """Configuration for the Lucas–Kanade flow."""
    xy_levels:      Field[int]   = Field(3, min=1, max=5, description="Mip levels of the x/y solve, coarse to fine; more follows faster motion")
    xy_regularizer: Field[float] = Field(0.001, min=0.0, max=0.1, step=0.0005, description="Damps the x/y solve where the window has no texture")
    xy_window:      Field[int]   = Field(2, min=1, max=4, description="x/y window radius in texels per level (2 = 5x5)", newline=True)
    xy_threshold:   Field[float] = Field(0.5, min=0.0, max=5.0, step=0.05, description="Displacement below this is noise, taken off the rest (px per frame)")
    depth_window:    Field[int]   = Field(1, min=0, max=4, description="Depth window radius in depth samples (1 = 3x3)", newline=True)
    depth_time:      Field[float] = Field(50.0, min=0.0, max=1000.0, description="Time over which the depth track averages (ms); 0 is the change of two frames")
    depth_threshold: Field[float] = Field(0.5, min=0.0, max=10.0, step=0.05, description="Depth motion below this is noise, taken off the rest (px per frame)")
    depth_tolerance: Field[float] = Field(0.1, min=0.01, max=1.0, description="Depth further than this is another surface: not averaged in, the track starts over (m)")
    strength_x:  Field[float] = Field(1.0, min=-20.0, max=20.0, description="Velocity per image width per second, x (negative inverts)", newline=True)
    strength_y:  Field[float] = Field(1.0, min=-20.0, max=20.0, description="Velocity per image width per second, y (negative inverts)")
    strength_z:  Field[float] = Field(0.0, min=0.0, max=10.0, description="Velocity per view width per second toward the camera; 1 balances it with x/y, 0 is no depth")
    use_mask:    Field[bool]  = Field(False, description="Multiply the velocity by the mask", newline=True)


class LucasKanadeFlow:
    """The 3D velocity (RGBA16F, xyz) between the newest two frames of its input."""

    def __init__(self, settings: LucasKanadeFlowSettings) -> None:
        self.settings: LucasKanadeFlowSettings = settings
        self._depth_velocity: DepthVelocity = DepthVelocity()
        self._mask: Texture | None = None

        self._frames: LuminanceFrames = LuminanceFrames()
        self._solve: SwapFbo = SwapFbo()            # the displacement as solved, level by level
        self._displacement: Fbo = Fbo()             # with the noise floor taken off
        self._velocity: Fbo = Fbo()
        self._shader: LucasKanadeShader = LucasKanadeShader()
        self._threshold_shader: DisplacementThreshold = DisplacementThreshold()
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
        """Its z, before `strength_z` and the mask."""
        return self._depth_velocity

    def allocate(self, width: int, height: int) -> None:
        self._depth_velocity.allocate(width, height)
        self._frames.allocate(width, height)
        self._solve.allocate(width, height, GL_RG32F)
        self._displacement.allocate(width, height, GL_RG32F)
        self._velocity.allocate(width, height, GL_RGBA16F)
        self._shader.allocate()
        self._threshold_shader.allocate()
        self._compose_shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._depth_velocity.deallocate()
        self._frames.deallocate()
        self._solve.deallocate()
        self._displacement.deallocate()
        self._velocity.deallocate()
        self._shader.deallocate()
        self._threshold_shader.deallocate()
        self._compose_shader.deallocate()

    def reset(self) -> None:
        self._depth_velocity.reset()
        self._frames.reset()
        self._solve.clear_all()
        self._displacement.clear()
        self._velocity.clear()

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._frames.set_input(texture)

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
        if not self._frames.ready:
            FlowUtil.zero(self._velocity)
            return
        s: LucasKanadeFlowSettings = self.settings

        self._solve.clear()                                 # the coarsest level starts from no motion
        for level in reversed(range(s.xy_levels)):
            self._solve.swap()
            self._solve.begin()
            self._shader.use(self._frames.previous_luminance, self._frames.luminance, self._solve.back_texture, level, s.xy_window, s.xy_regularizer)
            self._solve.end()

        self._displacement.begin()
        self._threshold_shader.use(self._solve.texture, s.xy_threshold)
        self._displacement.end()

        if depth is not None:
            self._depth_velocity.update_depth(depth, delta_time, view_span, depth_step, s.depth_window, s.depth_time,
                                              s.depth_threshold, s.depth_tolerance, self._displacement)

        # uv per frame as image widths per second: v is uv of the height, so a square motion stays square;
        # z is in view widths per frame already
        per_second: float = 1.0 / max(delta_time, 1e-6)
        aspect: float = self._displacement.height / max(self._displacement.width, 1)
        mask: Texture | None = self._mask if s.use_mask and self._mask is not None and self._mask.allocated else None
        self._velocity.begin()
        self._compose_shader.use(self._displacement, self._depth_velocity.depth_velocity if depth is not None else None,
                                 (s.strength_x * per_second, s.strength_y * per_second * aspect, s.strength_z * per_second), mask)
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


def lucas_kanade(previous: np.ndarray, current: np.ndarray, levels: int = 3, window: int = 2, regularizer: float = 0.001) -> np.ndarray:
    """The displacement (H, W, 2) in pixels per frame, x along columns and y along rows, that carries
    `previous` onto `current`: coarse to fine over `levels` mip levels, one solve per level over a
    (2·window+1)² window, the previous frame warped by the estimate so far. Mirrors the shader."""
    h, w = current.shape
    previous_levels: list[np.ndarray] = _pyramid(previous.astype(np.float64), levels)
    current_levels: list[np.ndarray] = _pyramid(current.astype(np.float64), levels)
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = (columns + 0.5) / w, (rows + 0.5) / h            # each output pixel's uv
    d: np.ndarray = np.zeros((h, w, 2))                     # the displacement in uv
    for level in reversed(range(levels)):
        p_image, c_image = previous_levels[level], current_levels[level]
        lh, lw = c_image.shape
        sxx = sxy = syy = sxt = syt = np.zeros((h, w))
        for j in range(-window, window + 1):
            for i in range(-window, window + 1):
                px, py = u * lw - 0.5 + i, v * lh - 0.5 + j
                qx, qy = px - d[..., 0] * lw, py - d[..., 1] * lh
                gx = 0.25 * (_bilinear(c_image, px + 1, py) - _bilinear(c_image, px - 1, py)
                             + _bilinear(p_image, qx + 1, qy) - _bilinear(p_image, qx - 1, qy))
                gy = 0.25 * (_bilinear(c_image, px, py + 1) - _bilinear(c_image, px, py - 1)
                             + _bilinear(p_image, qx, qy + 1) - _bilinear(p_image, qx, qy - 1))
                gt = _bilinear(c_image, px, py) - _bilinear(p_image, qx, qy)
                sxx, sxy, syy = sxx + gx * gx, sxy + gx * gy, syy + gy * gy
                sxt, syt = sxt + gx * gt, syt + gy * gt
        sxx, syy = sxx + regularizer, syy + regularizer
        det: np.ndarray = sxx * syy - sxy * sxy
        safe: np.ndarray = np.where(np.abs(det) > 1e-12, det, 1.0)
        dx = np.where(np.abs(det) > 1e-12, -(syy * sxt - sxy * syt) / safe, 0.0)
        dy = np.where(np.abs(det) > 1e-12, -(sxx * syt - sxy * sxt) / safe, 0.0)
        d = d + np.stack((dx / lw, dy / lh), axis=-1)
    return d * np.array([w, h])


def threshold_displacement(displacement: np.ndarray, threshold: float) -> np.ndarray:
    """A displacement (..., 2) in pixels with `threshold` pixels taken off its length, 0 below it. Mirrors displacement_threshold.frag."""
    length: np.ndarray = np.linalg.norm(displacement, axis=-1, keepdims=True)
    kept: np.ndarray = np.maximum(length - threshold, 0.0)
    return np.divide(displacement * kept, length, out=np.zeros_like(displacement), where=length > 1e-12)


def displacement_velocity(displacement: np.ndarray, dt: float, width: int, height: int, strength: tuple[float, float]) -> np.ndarray:
    """A displacement in uv per frame (..., 2) as a velocity in image widths per second, times `strength`.

    v is uv of the height, so it is taken to widths first: a square motion stays square.
    """
    return displacement * np.array([1.0, height / width]) / dt * np.array(strength)
