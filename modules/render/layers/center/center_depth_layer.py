"""The camera's depth in the center view, cut out once: the metres for computing, and a picture of them."""

# Third-party imports
from OpenGL.GL import GL_NEAREST, GL_R32F, GL_RGBA16F

# Local application imports
from modules.gl import Fbo, Style, Texture
from modules.settings import BaseSettings, Field
from ..layer_base import LayerBase
from ..source import DepthSourceLayer, DepthSourceLayerSettings
from .center_geometry import CenterGeometry, SourceSpace
from ...shaders import DepthView, DrawRoi, MaskApply


class CenterDepthLayerSettings(BaseSettings):
    """How the centered depth's picture is masked."""
    use_mask:     Field[bool]  = Field(True, description="The mask as the picture's alpha")
    mask_opacity: Field[float] = Field(1.0, min=0.0, max=1.0, description="Mask alpha strength")


class CenterDepthLayer(LayerBase):
    """The camera's depth (`DepthSourceLayer.depth`) cut into the center view once, NEAREST so an edge
    never reads as a distance in between. `depth` is the metres, unmasked, for computing; `texture`
    is a picture of them (`DepthView`, with the depth source's draw settings), the mask as alpha.
    `image` is the layer the picture lies over in `DepthDrawMode.OVER_IMAGE`: the centered camera image,
    read at each draw since a layer's output texture can change."""

    def __init__(self, cam_id: int, geometry: CenterGeometry, depth: DepthSourceLayer, image: LayerBase, mask: Texture,
                 settings: CenterDepthLayerSettings, view_settings: DepthSourceLayerSettings) -> None:
        self._cam_id: int = cam_id
        self._geometry: CenterGeometry = geometry
        self._depth_source: DepthSourceLayer = depth
        self._image: LayerBase = image
        self._mask: Texture = mask
        self.settings: CenterDepthLayerSettings = settings
        self._view_settings: DepthSourceLayerSettings = view_settings
        self._depth: Fbo = Fbo(interpolation=GL_NEAREST)
        self._view: Fbo = Fbo()
        self._masked: Fbo = Fbo()
        self._roi_shader: DrawRoi = DrawRoi()
        self._view_shader: DepthView = DepthView()
        self._mask_shader: MaskApply = MaskApply()
        self._output: Fbo = self._view

    @property
    def depth(self) -> Texture:
        """Metres in red, unmasked; 0 where there is no depth."""
        return self._depth

    @property
    def texture(self) -> Texture:
        return self._output

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._depth.allocate(width, height, GL_R32F)
        self._view.allocate(width, height, GL_RGBA16F)
        self._masked.allocate(width, height, GL_RGBA16F)
        self._roi_shader.allocate()
        self._view_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._depth.deallocate()
        self._view.deallocate()
        self._masked.deallocate()
        self._roi_shader.deallocate()
        self._view_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        if self._geometry.lost:
            self._depth.clear()
            self._view.clear()
            self._masked.clear()
            return
        source: Texture = self._depth_source.depth
        if not self._geometry.ready or not source.allocated:
            return

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._depth.clear()             # outside the source is no depth, not the last frame's
        self._depth.begin()
        self._roi_shader.use(source, *self._geometry.source_roi(SourceSpace.IMAGE, source))
        self._depth.end()

        view = self._view_settings
        self._view.clear()
        self._view.begin()
        self._view_shader.use(self._depth, self._image.texture, view.near, view.far, view.draw_mode)
        self._view.end()

        self._output = self._view
        if self.settings.use_mask and self._mask.allocated:
            self._masked.clear()
            self._masked.begin()
            self._mask_shader.use(self._view, self._mask, self.settings.mask_opacity)
            self._masked.end()
            self._output = self._masked

        Style.pop_style()
