"""Lucas–Kanade: the displacement of each pixel between two luminance frames, coarse to fine.

Where `OpticalFlow` gives a motion strength per pixel (the brightness change along the gradient),
this gives how far the image moved: a window of gradients per pixel, solved for the shift that
carries the previous frame onto the current one, first on the coarsest mip level, then refined once
per finer level with the previous frame warped by the estimate so far. The result is a displacement
in uv per frame, which `DepthVelocity` follows to measure depth along the motion, and a velocity for
the bridges, shaped as `OpticalFlow` shapes its own (`strength`, `threshold`, `boost`).

`lucas_kanade` below is the numpy reference of the shader (lucas_kanade_shader.frag).
"""

import numpy as np
from OpenGL.GL import GL_RG16F, GL_RG32F

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field

from .optical_flow import OpticalFlowSettings
from .shaders import DisplacementVelocity, LucasKanadeShader


class LucasKanadeSettings(BaseSettings):
    """Configuration for the Lucas–Kanade displacement."""
    levels:             Field[int]   = Field(3, min=1, max=5, description="Mip levels, coarse to fine; more follows faster motion")
    window:             Field[int]   = Field(2, min=1, max=4, description="Window radius in texels per level (2 = 5x5)")
    regularizer:        Field[float] = Field(0.001, min=0.0, max=0.1, step=0.0005, description="Damps the solve where the window has no texture")
    displacement_scale: Field[float] = Field(0.1, min=0.0, max=2.0, step=0.01, description="Flow velocity per pixel of displacement per frame")


class LucasKanade:
    """The displacement (uv per frame, RG32F) of the current luminance frame from the previous one,
    and its velocity for the bridges (RG16F), both at the flow's size."""

    def __init__(self, settings: LucasKanadeSettings) -> None:
        self.settings: LucasKanadeSettings = settings
        self._displacement: SwapFbo = SwapFbo()
        self._velocity: Fbo = Fbo()
        self._shader: LucasKanadeShader = LucasKanadeShader()
        self._velocity_shader: DisplacementVelocity = DisplacementVelocity()

    @property
    def displacement(self) -> Texture:
        """How far each pixel moved since the previous frame, in uv."""
        return self._displacement.texture

    @property
    def velocity(self) -> Texture:
        """The displacement shaped for the bridges."""
        return self._velocity

    def allocate(self, width: int, height: int) -> None:
        self._displacement.allocate(width, height, GL_RG32F)
        self._velocity.allocate(width, height, GL_RG16F)
        self._shader.allocate()
        self._velocity_shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._displacement.deallocate()
        self._velocity.deallocate()
        self._shader.deallocate()
        self._velocity_shader.deallocate()

    def reset(self) -> None:
        self._displacement.clear_all()
        self._velocity.clear()

    def update(self, previous: Texture, current: Texture, shaping: OpticalFlowSettings) -> None:
        """Measure the displacement from `previous` to `current` (mipmapped luminance), then shape it."""
        s: LucasKanadeSettings = self.settings
        self._displacement.clear()                          # the coarsest level starts from no motion
        for level in reversed(range(s.levels)):
            self._displacement.swap()
            self._displacement.begin()
            self._shader.use(previous, current, self._displacement.back_texture, level, s.window, s.regularizer)
            self._displacement.end()

        self._velocity.begin()
        self._velocity_shader.use(self._displacement.texture, s.displacement_scale, shaping.strength_x, shaping.strength_y,
                                  shaping.threshold, 1.0 - shaping.boost)
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
