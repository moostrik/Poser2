"""The capture's monitor: the capture's color, its depth in a colormap, or the depth over the color."""

from OpenGL.GL import GL_RGBA16F

from modules import render
from modules.gl import Fbo, Style, Texture, clear_color
from ....settings import CaptureSettings, CaptureView, ViewCropSettings

DRAW_MODES: dict[CaptureView, render.DepthDrawMode] = {CaptureView.DEPTH: render.DepthDrawMode.COLOR,
                                                       CaptureView.BOTH: render.DepthDrawMode.OVER_IMAGE}


class CaptureMonitorLayer(render.LayerBase):
    """The capture's `image` and `depth` as `settings.show` says, the depth on a log scale from the view's `near` to
    its `far` (`band`, the view's crop, its front and back); drawn again with each new capture."""

    def __init__(self, image: render.CaptureImageLayer, depth: render.CaptureDepthLayer, settings: CaptureSettings,
                 band: ViewCropSettings) -> None:
        self._image: render.CaptureImageLayer = image
        self._depth: render.CaptureDepthLayer = depth
        self.settings: CaptureSettings = settings
        self._band: ViewCropSettings = band
        self._fbo: Fbo = Fbo()
        self._depth_colormap: render.DepthColormap = render.DepthColormap()
        self._drawn: tuple | None = None

    @property
    def texture(self) -> Texture:
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._fbo.allocate(width, height, GL_RGBA16F)
        self._depth_colormap.allocate()
        self._drawn = None

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._depth_colormap.deallocate()

    def update(self) -> None:
        s: CaptureSettings = self.settings
        show: CaptureView = CaptureView(s.show)
        near, far = self._band.near, self._band.far
        drawn: tuple = (self._image.timestamp, show, s.colormap, near, far)
        if drawn == self._drawn or not self._fbo.allocated:
            return
        self._drawn = drawn
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._fbo.begin()
        clear_color()
        if show == CaptureView.COLOR:
            if self._image.texture.allocated:
                self._image.texture.draw()
        else:
            self._depth_colormap.use(self._depth.texture, self._image.texture, near, far, DRAW_MODES[show], s.colormap)
        self._fbo.end()
        Style.pop_style()
