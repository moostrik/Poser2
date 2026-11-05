# Third-party imports
from OpenGL.GL import GL_R16F

# Local application imports
from modules.gl import Fbo, Style, Texture
from ..layer_base import LayerBase
from .depth_source_layer import DepthSourceLayer
from ...shaders import DepthMask


class DepthMaskLayer(LayerBase):
    """The player cut from the capture's depth: 1 where the depth lies from the depth source's `near`
    to `far`, 0 outside or without depth. On the view of the camera image, at the depth's size, read
    linearly, so the depth's few rows soften instead of stepping. Cut again with each depth frame,
    told by its capture time, and when `near` or `far` change."""

    def __init__(self, depth: DepthSourceLayer) -> None:
        self._depth: DepthSourceLayer = depth
        self._fbo: Fbo = Fbo()
        self._shader: DepthMask = DepthMask()
        self._timestamp: float | None = None     # the capture time of the depth frame the mask is cut from
        self._range: tuple[float, float] | None = None     # the near and far it was cut with

    @property
    def texture(self) -> Texture:
        return self._fbo

    @property
    def timestamp(self) -> float | None:
        """The capture time of the depth frame the mask it holds is cut from, the depth source's; None before the first."""
        return self._timestamp

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        self._shader.allocate()     # the FBO on the first depth, at its size

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        timestamp: float | None = self._depth.timestamp
        near_far: tuple[float, float] = (self._depth.settings.near, self._depth.settings.far)
        if timestamp is None or (timestamp == self._timestamp and near_far == self._range):
            return
        depth: Texture = self._depth.depth
        if not self._fbo.allocated:
            self._fbo.allocate(depth.width, depth.height, GL_R16F)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.begin()
        self._shader.use(depth, *near_far)
        self._fbo.end()
        Style.pop_style()
        self._timestamp = timestamp
        self._range = near_far
