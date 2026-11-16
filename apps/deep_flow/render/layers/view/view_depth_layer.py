"""The view stage's depth as a picture, for the panels and the show."""

# Third-party imports
from OpenGL.GL import GL_RGBA16F

# Local application imports
from modules.gl import Fbo, Style, Texture, shaders
from modules.render import DepthColormap, LayerBase
from ....settings import DepthDrawMode, ViewDepthLayerSettings
from .view_layer import ViewLayer


class ViewDepthLayer(LayerBase):
    """The view's depth as a picture (`DepthColormap`), over the view's image in `DepthDrawMode.OVER_IMAGE`, the mask as
    alpha; drawn again with each new view."""

    def __init__(self, view: ViewLayer, settings: ViewDepthLayerSettings) -> None:
        self._view: ViewLayer = view
        self.settings: ViewDepthLayerSettings = settings
        self._picture: Fbo = Fbo()
        self._masked: Fbo = Fbo()
        self._picture_shader: DepthColormap = DepthColormap()
        self._mask_shader: shaders.MaskApply = shaders.MaskApply()
        self._output: Fbo = self._picture
        self._drawn: tuple | None = None

    @property
    def texture(self) -> Texture:
        return self._output

    @property
    def timestamp(self) -> float | None:
        return self._view.timestamp

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._picture.allocate(width, height, GL_RGBA16F)
        self._masked.allocate(width, height, GL_RGBA16F)
        self._picture_shader.allocate()
        self._mask_shader.allocate()
        self._drawn = None

    def deallocate(self) -> None:
        self._picture.deallocate()
        self._masked.deallocate()
        self._picture_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        s: ViewDepthLayerSettings = self.settings
        output = self._view.output
        drawn: tuple = (self._view.timestamp, output.depth.near, output.depth.far, s.draw_mode, s.colormap, s.use_mask,
                        s.mask_opacity)
        if drawn == self._drawn:
            return
        self._drawn = drawn
        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        self._picture.clear()
        self._picture.begin()
        self._picture_shader.use(output.depth.texture, output.image, output.depth.near, output.depth.far, s.draw_mode,
                                 s.colormap)
        self._picture.end()
        self._output = self._picture
        if s.use_mask:
            self._masked.clear()
            self._masked.begin()
            self._mask_shader.use(self._picture, output.mask, s.mask_opacity)
            self._masked.end()
            self._output = self._masked
        Style.pop_style()
