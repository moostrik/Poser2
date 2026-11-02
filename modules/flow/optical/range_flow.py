"""Range flow: the motion of each pixel from depth alone, sideways and toward the camera in one equation.

A surface point at pixel x now was at x − d a frame ago, and its depth changed by W meanwhile, so
    Zx·u + Zy·v − W = −Zt
with Z the smoothed depth, (u, v) the remaining displacement and Zt the depth change at x after the
estimate so far. Each pixel solves it by least squares over a (2·window+1)² window, coarse to fine
over sampling strides 4, 2, 1 texels (depth gets no mip levels: averaging would mix a player's
distance with the wall's). Window samples whose depth jumps more than `jump` from the center pixel's,
or between frames, are left out, so a window never mixes two surfaces at a silhouette. A small
regularizer pulls (u, v) toward 0 where the depth is flat. On a flat slope sliding sideways and moving
in depth cannot be told apart: range flow needs curvature, as a body has.

The result is RGBA32F: the displacement in uv per frame (RG), the velocity toward the camera in m/s
(B, −W/dt), and the share of the window that counted (A). `range_flow` below is the numpy reference of
the shader (range_flow_shader.frag).
"""

import numpy as np
from OpenGL.GL import GL_RG16F, GL_RGBA32F

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field

from .optical_flow import OpticalFlowSettings
from .shaders import DisplacementVelocity, RangeFlowShader


class RangeFlowSettings(BaseSettings):
    """Configuration for range flow."""
    levels:             Field[int]   = Field(3, min=1, max=4, description="Sampling strides 2^(levels-1) down to 1 texel; more follows faster motion")
    window:             Field[int]   = Field(2, min=1, max=4, description="Window radius in strides (2 = 5x5)")
    regularizer:        Field[float] = Field(0.0001, min=0.0, max=0.01, step=0.00005, description="Damps sideways motion where the depth is flat (m² per texel²)")
    displacement_scale: Field[float] = Field(0.1, min=0.0, max=2.0, step=0.01, description="Flow velocity per pixel of displacement per frame")


class RangeFlow:
    """The motion from two smoothed depths: `motion` (RGBA32F, see the module docstring) and its
    sideways part as a velocity for the bridges (RG16F), both at the flow's size."""

    def __init__(self, settings: RangeFlowSettings) -> None:
        self.settings: RangeFlowSettings = settings
        self._motion: SwapFbo = SwapFbo()
        self._velocity: Fbo = Fbo()
        self._shader: RangeFlowShader = RangeFlowShader()
        self._velocity_shader: DisplacementVelocity = DisplacementVelocity()

    @property
    def motion(self) -> Texture:
        """Displacement in uv per frame (RG), velocity toward the camera in m/s (B), the share of the window that counted (A)."""
        return self._motion.texture

    @property
    def velocity(self) -> Texture:
        """The displacement shaped for the bridges."""
        return self._velocity

    def allocate(self, width: int, height: int) -> None:
        self._motion.allocate(width, height, GL_RGBA32F)
        self._velocity.allocate(width, height, GL_RG16F)
        self._shader.allocate()
        self._velocity_shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._motion.deallocate()
        self._velocity.deallocate()
        self._shader.deallocate()
        self._velocity_shader.deallocate()

    def reset(self) -> None:
        self._motion.clear_all()
        self._velocity.clear()

    def update(self, previous: Texture, current: Texture, delta_time: float, jump: float, shaping: OpticalFlowSettings) -> None:
        """Measure the motion from `previous` to `current` (smoothed depth in metres), then shape its sideways part."""
        s: RangeFlowSettings = self.settings
        self._motion.clear()                                # the coarsest stride starts from no motion
        for level in reversed(range(s.levels)):
            self._motion.swap()
            self._motion.begin()
            self._shader.use(previous, current, self._motion.back_texture, 2 ** level, s.window, s.regularizer, jump, delta_time)
            self._motion.end()

        self._velocity.begin()
        self._velocity_shader.use(self._motion.texture, s.displacement_scale, shaping.strength_x, shaping.strength_y,
                                  shaping.threshold, 1.0 - shaping.boost)
        self._velocity.end()


# ── numpy reference ─────────────────────────────────────────────────────

def _nearest(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`image` at pixel coordinates (centers at integers), nearest, clamped: what `texelFetch` reads."""
    h, w = image.shape
    return image[np.clip(np.floor(y + 0.5).astype(int), 0, h - 1), np.clip(np.floor(x + 0.5).astype(int), 0, w - 1)]


def range_flow(previous: np.ndarray, current: np.ndarray, delta_time: float, levels: int = 3, window: int = 2,
               regularizer: float = 0.0001, jump: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """The displacement (H, W, 2) in pixels per frame (x along columns, y along rows) and the velocity
    toward the camera (H, W) in m/s, from two depths in metres (0 for none). Mirrors the shader."""
    h, w = current.shape
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    d: np.ndarray = np.zeros((h, w, 2))                     # pixels per frame
    toward: np.ndarray = np.zeros((h, w))
    center: np.ndarray = current
    for level in reversed(range(levels)):
        stride: int = 2 ** level
        a = {key: np.zeros((h, w)) for key in ('uu', 'uv', 'vv', 'uw', 'vw', 'ww', 'bu', 'bv', 'bw')}
        count: np.ndarray = np.zeros((h, w))
        for j in range(-window, window + 1):
            for i in range(-window, window + 1):
                px, py = columns + i * stride, rows + j * stride
                z1 = _nearest(current, px, py)
                z0 = _nearest(previous, px - d[..., 0], py - d[..., 1])
                right, left = _nearest(current, px + stride, py), _nearest(current, px - stride, py)
                down, up = _nearest(current, px, py + stride), _nearest(current, px, py - stride)
                valid = ((z1 > 0) & (z0 > 0) & (right > 0) & (left > 0) & (down > 0) & (up > 0) & (center > 0)
                         & (np.abs(z1 - center) <= jump) & (np.abs(z1 - z0) <= jump)
                         & (np.abs(right - z1) <= jump) & (np.abs(left - z1) <= jump)
                         & (np.abs(down - z1) <= jump) & (np.abs(up - z1) <= jump))
                gx = np.where(valid, (right - left) / (2.0 * stride), 0.0)
                gy = np.where(valid, (down - up) / (2.0 * stride), 0.0)
                gt = np.where(valid, z1 - z0, 0.0)
                one = valid.astype(np.float64)
                # rows of [gx, gy, -1] · [du, dv, W] = -gt
                a['uu'] += gx * gx; a['uv'] += gx * gy; a['vv'] += gy * gy
                a['uw'] -= gx; a['vw'] -= gy; a['ww'] += one
                a['bu'] -= gx * gt; a['bv'] -= gy * gt; a['bw'] += gt
                count += one
        m = np.stack([np.stack([a['uu'] + regularizer, a['uv'], a['uw']], -1),
                      np.stack([a['uv'], a['vv'] + regularizer, a['vw']], -1),
                      np.stack([a['uw'], a['vw'], a['ww'] + 1e-6], -1)], -2)
        b = np.stack([a['bu'], a['bv'], a['bw']], -1)
        enough: np.ndarray = count >= 6
        solution: np.ndarray = np.zeros((h, w, 3))
        solution[enough] = np.linalg.solve(m[enough], b[enough][..., None])[..., 0]
        d = d + solution[..., :2]
        toward = np.where(enough, -solution[..., 2] / delta_time, 0.0)
    return d, toward
