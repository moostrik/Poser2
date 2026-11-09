"""Lucas–Kanade Flow: how far each pixel moved between two luminance frames, coarse to fine; the x/y solve.

Where the gradient flow gives a motion strength per pixel (the brightness change along the gradient),
this gives how far the image moved: a window of gradients per pixel, solved for the shift that carries
the previous frame onto the current one, first on the coarsest mip level, then refined once per finer
level with the previous frame warped by the estimate so far. Each level is solved at its own size, on
its own texels, and starts from the coarser level's displacement read linear (Bouguet's pyramidal
Lucas–Kanade); solving every full-size pixel at a coarse level instead puts each window a fraction of a
texel off its neighbour's, and where a window is badly conditioned that flips the solve from pixel to
pixel, in rows and columns.

Out comes the displacement in uv per frame: `solved`, as solved, and `displacement`, with `threshold`
(px per frame) taken off its length, as below it is the solve's jitter, not motion. With a depth and
`same_surface`, only the window's samples within a tolerance of the centre's depth count.
`LucasKanadeFlow3D` composes this solve with the depth velocity into a 3D velocity.

The settings in flow pixels go with the flow's resolution: `levels` (each level doubles the motion it
follows, about 2 × 2^(levels − 1) px per frame at full size, so halving the resolution needs one level
fewer for the same motion), `window` and `threshold`.

`lucas_kanade` and `threshold_displacement` below are the numpy references of the shaders.
"""

import numpy as np
from OpenGL.GL import GL_RG32F

from modules.gl import Fbo, SwapFbo, Texture
from modules.settings import BaseSettings, Field
from ..port import Port, TextureKind
from .luminance_frames import LuminanceFrames
from .patch import Patch
from .shaders import DisplacementThreshold, LucasKanadeShader

MAX_LEVELS: int = 9     # the most pyramid levels a solve has an Fbo for


class LucasKanadeFlowSettings(BaseSettings):
    """The x/y solve of the displacement."""
    levels:         Field[int]   = Field(3, min=1, max=5, description="Pyramid levels of the x/y solve, each doubles the reach (more: faster motion, more spill)")
    window:         Field[Patch] = Field(Patch.SEVEN, min=Patch.THREE, max=Patch.NINE, description="x/y patch size (larger: cleaner, more spill)")
    damping:        Field[float] = Field(0.03, min=0.0, max=0.1, step=0.001, description="Damps x/y where the patch has little texture (higher: less noise, smaller motion)")
    same_surface:   Field[bool]  = Field(True, newline=True, width=48, description="Patch counts only the center's surface (on: less smear, wide fast parts slower)")
    threshold:      Field[float] = Field(0.5, min=0.0, max=5.0, step=0.05, description="x/y below this is noise, in px/frame (higher: less noise, less slow motion)")


class LucasKanadeFlow:
    """The displacement (RG32F, uv per frame) between the newest two frames of its input."""

    def __init__(self, settings: LucasKanadeFlowSettings) -> None:
        self.settings: LucasKanadeFlowSettings = settings
        self._frames: LuminanceFrames = LuminanceFrames()
        self._solve: SwapFbo = SwapFbo()            # the displacement as solved, at full size: level 0
        self._levels: list[SwapFbo] = [SwapFbo() for _ in range(1, MAX_LEVELS)]    # levels 1 and up, each at its own size
        self._displacement: Fbo = Fbo()             # with the noise floor taken off
        self._shader: LucasKanadeShader = LucasKanadeShader()
        self._threshold_shader: DisplacementThreshold = DisplacementThreshold()

    @property
    def ready(self) -> bool:
        """Whether it holds two frames to measure between."""
        return self._frames.ready

    @property
    def luminance(self) -> Texture:
        """The newest luminance frame."""
        return self._frames.luminance

    @property
    def previous_luminance(self) -> Texture:
        """The luminance frame before the newest."""
        return self._frames.previous_luminance

    @property
    def solved(self) -> Texture:
        """The displacement as solved (RG32F, uv per frame), before the threshold."""
        return self._solve

    @property
    def displacement(self) -> Texture:
        """The displacement (RG32F, uv per frame) with `threshold` taken off its length."""
        return self._displacement

    @property
    def ports(self) -> dict[str, Port]:
        """The luminance it measures on, the pyramid levels it solves coarse to fine, the solve and the displacement."""
        used: int = min(self.settings.levels, MAX_LEVELS) - 1      # the levels above the full-size solve
        levels: dict[str, Port] = {f'level_{level}': Port(self._levels[level - 1], TextureKind.VELOCITY)
                                   for level in range(used, 0, -1)}
        return {'luminance': Port(self._frames.luminance, TextureKind.IMAGE), **levels,
                'solve': Port(self._solve, TextureKind.VELOCITY),
                'displacement': Port(self._displacement, TextureKind.VELOCITY)}

    def allocate(self, width: int, height: int) -> None:
        self._frames.allocate(width, height)
        self._solve.allocate(width, height, GL_RG32F)
        for level, fbo in enumerate(self._levels, start=1):
            fbo.allocate(max(width >> level, 1), max(height >> level, 1), GL_RG32F)     # the size of the frames' mip level
        self._displacement.allocate(width, height, GL_RG32F)
        self._shader.allocate()
        self._threshold_shader.allocate()
        self.reset()

    def deallocate(self) -> None:
        self._frames.deallocate()
        self._solve.deallocate()
        for fbo in self._levels:
            fbo.deallocate()
        self._displacement.deallocate()
        self._shader.deallocate()
        self._threshold_shader.deallocate()

    def reset(self) -> None:
        self._frames.reset()
        self._solve.clear_all()
        for fbo in self._levels:
            fbo.clear_all()
        self._displacement.clear()

    def set_input(self, texture: Texture) -> None:
        """Take `texture` as the newest frame."""
        self._frames.set_input(texture)

    def update(self, depth: Texture | None = None, tolerance: float = 0.0) -> None:
        """Solve the displacement between the newest two frames; with `depth` (metres in red) and
        `same_surface`, a window counts only the samples within `tolerance` of its centre's depth."""
        if not self._frames.ready:
            self._solve.clear_all()
            self._displacement.clear()
            return
        s: LucasKanadeFlowSettings = self.settings
        self._pyramid(self._solve, self._frames.previous_luminance, self._frames.luminance, depth if s.same_surface else None, tolerance)

        self._displacement.begin()
        self._threshold_shader.use(self._solve, s.threshold)
        self._displacement.end()

    def _pyramid(self, solve: SwapFbo, previous: Texture, current: Texture, depth: Texture | None, tolerance: float) -> None:
        """The displacement carrying `previous` onto `current`, coarse to fine, into `solve`: one solve
        per level at that level's size, each from the coarser one's displacement read linear (uv, so
        it needs no scaling). The finest is `solve`."""
        s: LucasKanadeFlowSettings = self.settings
        estimate: Texture | None = None
        for level in reversed(range(min(s.levels, MAX_LEVELS))):
            target: SwapFbo = solve if level == 0 else self._levels[level - 1]
            target.swap()
            target.begin()
            self._shader.use(previous, current, estimate, level, s.window, s.damping, depth, tolerance)     # the coarsest from no motion
            target.end()
            estimate = target


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
    reach: float = float(window + 1)                         # a level resolves at most its window's reach
    dx = np.clip(np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0), -reach, reach)
    dy = np.clip(np.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0), -reach, reach)
    return d + np.stack((dx / lw, dy / lh), axis=-1)


def lucas_kanade(previous: np.ndarray, current: np.ndarray, levels: int = 3, window: int = 2, damping: float = 0.001,
                 depth: np.ndarray | None = None, tolerance: float = 0.0) -> np.ndarray:
    """The displacement (H, W, 2) in pixels per frame, x along columns and y along rows, that carries
    `previous` onto `current`: coarse to fine over `levels` mip levels, one solve per level over a
    (2·window+1)² window, each level solved at its own size from the coarser one's displacement read
    linear, the previous frame warped by it. With `depth` (H, W, m), only the window's samples within
    `tolerance` of the centre's depth count. Mirrors the shaders."""
    h, w = current.shape
    previous_levels: list[np.ndarray] = _pyramid(previous.astype(np.float64), levels)
    current_levels: list[np.ndarray] = _pyramid(current.astype(np.float64), levels)
    d: np.ndarray | None = None                             # the displacement in uv, at the coarser level's size
    for level in reversed(range(levels)):
        lh, lw = current_levels[level].shape
        rows, columns = np.mgrid[0:lh, 0:lw].astype(np.float64)
        u, v = (columns + 0.5) / lw, (rows + 0.5) / lh      # this level's texel centers in uv
        if d is None:
            start: np.ndarray = np.zeros((lh, lw, 2))
        else:
            ch, cw = d.shape[:2]
            start = np.stack([_bilinear(d[..., c], u * cw - 0.5, v * ch - 0.5) for c in range(2)], axis=-1)
        d = _solve_level(previous_levels[level], current_levels[level], start, u, v, window, damping, depth, tolerance)
    assert d is not None
    return d * np.array([w, h])


def threshold_displacement(displacement: np.ndarray, threshold: float) -> np.ndarray:
    """A displacement (..., 2) in pixels with `threshold` pixels taken off its length, 0 below it. Mirrors displacement_threshold.frag."""
    length: np.ndarray = np.linalg.norm(displacement, axis=-1, keepdims=True)
    kept: np.ndarray = np.maximum(length - threshold, 0.0)
    return np.divide(displacement * kept, length, out=np.zeros_like(displacement), where=length > 1e-12)
