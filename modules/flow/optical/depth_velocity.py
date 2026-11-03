"""Depth Velocity.

What the image cannot measure: the player's motion toward the camera, from the depth image. One
depth sample is mostly noise from frame to frame, so z is measured like the flow's x/y: over a
window of depth samples, the change of each along the motion. A neighbour is averaged in only when
its depth is within `tolerance` of the pixel's (not the background beside the player) and of its own
depth the frame before (not a limb passing over it). Where less than `coverage` of the window is
averaged in, as at a silhouette, z is 0: one sample alone is the noisiest reading there is. The change
comes in view widths per frame, over the metres one view width spans at the depth (`view_span` per
metre). Each optical flow owns one for its z, and turns it into view widths per second times its own
`strength_z`.

`window_depth_displacement` and `depth_threshold` below are the numpy references of the shaders.
"""
import numpy as np
from OpenGL.GL import GL_R16F, GL_R32F

from modules.gl import Fbo, Texture
from modules.settings import Field, BaseSettings
from .. import FlowBase, FlowUtil
from .shaders import DepthThreshold, DepthVelocityShader


class DepthVelocitySettings(BaseSettings):
    """Configuration for the depth velocity of a flow that has no depth settings of its own."""
    window = Field(2, min=0, max=4, description="Window radius in depth samples (2 = 5x5)")
    threshold = Field(0.5, min=0.0, max=10.0, step=0.05, description="Depth motion below this is noise, taken off the rest (px per frame)")
    tolerance = Field(0.1, min=0.01, max=1.0, description="Neighbours further than this in depth are not averaged in (m)")
    coverage = Field(0.5, min=0.0, max=1.0, step=0.05, description="Share of the depth window averaged in that z needs; less reads 0")


class DepthVelocity(FlowBase):
    """Signed motion toward the camera in view widths per frame (R16F), positive toward it.

    Pipeline, on each new flow frame:
        1. Take the depth → the input SwapFbo (R32F, metres, at the flow's size, nearest)
        2. Measure over the window against the last one → the difference (R16F)
        3. Take the noise floor off → output_fbo (R16F)
    """

    def __init__(self) -> None:
        super().__init__()

        self._input_internal_format = GL_R32F
        self._output_internal_format = GL_R16F

        self._velocity_shader: DepthVelocityShader = DepthVelocityShader()
        self._threshold_shader: DepthThreshold = DepthThreshold()

        self._difference: Fbo = Fbo()

    @property
    def depth_velocity(self) -> Texture:
        """Motion toward the camera in view widths per frame (R16F)."""
        return self._output

    @property
    def difference(self) -> Texture:
        """The motion toward the camera before the threshold (view widths per frame, R16F)."""
        return self._difference

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        super().allocate(width, height, output_width, output_height)
        self._difference.allocate(width, height, GL_R16F)
        FlowUtil.zero(self._difference)
        self._velocity_shader.allocate()
        self._threshold_shader.allocate()

    def deallocate(self) -> None:
        super().deallocate()
        self._difference.deallocate()
        self._velocity_shader.deallocate()
        self._threshold_shader.deallocate()

    def reset(self) -> None:
        super().reset()
        FlowUtil.zero(self._difference)

    def update_depth(self, depth: Texture, view_span: float, depth_step: tuple[float, float], window: int,
                     threshold: float, tolerance: float, coverage: float, displacement: Texture | None = None) -> None:
        """Advance on a new flow frame: measure against the last depth, then take the noise floor off.

        Args:
            depth: Metres in red, 0 where there is none
            view_span: Metres one view width spans per metre of depth; 0 or less reads no motion
            depth_step: uv between two depth samples in the flow's view; 0 takes one flow texel
            window: Radius in depth samples
            threshold: Depth motion below this is noise, taken off the rest (px per frame)
            tolerance: Neighbours further than this in depth are not averaged in (m)
            coverage: Share of the window averaged in that z needs; less reads 0
            displacement: How far each pixel moved since then (uv); when given, depth is measured
                along the motion (Lucas–Kanade flow)
        """
        if not self._allocated:
            return
        width, height = self._input_fbo.width, self._input_fbo.height
        step: tuple[float, float] = depth_step if depth_step[0] > 0.0 and depth_step[1] > 0.0 else (1.0 / width, 1.0 / height)

        self._input_fbo.swap()
        FlowUtil.blit(self._input_fbo, depth)

        self._difference.begin()
        self._velocity_shader.use(self._input_fbo.texture, self._input_fbo.back_texture, view_span, step, window, tolerance, coverage, displacement)
        self._difference.end()

        self._output_fbo.begin()
        self._threshold_shader.use(self._difference, threshold / max(width, 1))     # px as view widths
        self._output_fbo.end()


# ── numpy reference ─────────────────────────────────────────────────────

def window_depth_displacement(current: np.ndarray, previous: np.ndarray, displacement: np.ndarray, window: int,
                              step: tuple[float, float], tolerance: float, coverage: float, view_span: float) -> np.ndarray:
    """Motion toward the camera in view widths per frame (H, W), as the shader measures it.

    `current` and `previous` are depth in metres (H, W), 0 where there is none; `displacement` is how
    far each pixel moved (H, W, 2, pixels per frame, x along columns, y along rows); `step` is the
    pixels between two depth samples (x, y). Over the (2·window+1)² samples around a pixel, a sample
    is averaged in when it has depth now and where it came from, within `tolerance` of both that and
    the pixel's depth; the mean of their changes, over `depth × view_span`, is the motion. 0 without
    depth at the pixel, without a span, or with less than `coverage` of the window averaged in.
    """
    h, w = current.shape
    rows, columns = np.mgrid[0:h, 0:w].astype(np.float64)
    total: np.ndarray = np.zeros((h, w))
    kept: np.ndarray = np.zeros((h, w))
    measured: np.ndarray = (current > 0.0) & (view_span > 0.0)
    for j in range(-window, window + 1):
        for i in range(-window, window + 1):
            x, y = columns + 0.5 + i * step[0], rows + 0.5 + j * step[1]
            now: np.ndarray = current[np.clip(np.floor(y).astype(int), 0, h - 1), np.clip(np.floor(x).astype(int), 0, w - 1)]
            then: np.ndarray = previous[np.clip(np.floor(y - displacement[..., 1]).astype(int), 0, h - 1),
                                        np.clip(np.floor(x - displacement[..., 0]).astype(int), 0, w - 1)]
            change: np.ndarray = now - then
            keep: np.ndarray = measured & (now > 0.0) & (then > 0.0) & (np.abs(change) <= tolerance) & (np.abs(now - current) <= tolerance)
            total += np.where(keep, -change / np.where(keep, now * view_span, 1.0), 0.0)
            kept += keep
    needed: float = max(coverage * (2 * window + 1) ** 2, 1.0)
    return np.where(kept >= needed, total / np.maximum(kept, 1), 0.0)


def depth_threshold(value: np.ndarray, threshold: float) -> np.ndarray:
    """A signed field with `threshold` taken off its size, 0 below it: the noise floor goes, and what is above stays continuous."""
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)
