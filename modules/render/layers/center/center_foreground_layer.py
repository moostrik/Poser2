"""Renders centered and rotated foreground with the foreground's look (`ForegroundLook`) and optional mask."""

# Local application imports
from ..layer_base import LayerBase, HasCaptureTime
from ..generic import ForegroundLook, ForegroundLookSettings
from .center_geometry import CenterGeometry, SourceSpace
from ...shaders import DrawRoi, MaskMultiply
from modules.gl import Fbo, Texture, Style
from ...color_settings import ColorSettings


class CenterForegroundLayerSettings(ForegroundLookSettings):
    """Configuration for CenterForegroundLayer: the foreground's look."""


class CenterForegroundLayer(LayerBase):
    """Renders foreground image cropped and rotated, with the foreground's look.

    `space` says whether the foreground is on the pose's crop or on the camera image. With
    `premultiply` the source is a whole image, and the mask multiplies it right after the cut-out, so
    the look starts from the player's colors on black as from a masked source (RVM's is). Optionally
    applies mask texture as alpha for compositing. `source` is the layer whose texture it draws;
    `timestamp` is that frame's capture time, forwarded.
    """

    def __init__(self, cam_id: int, geometry: CenterGeometry, source: LayerBase, mask_texture: Texture, space: SourceSpace,
                 premultiply: bool, settings: CenterForegroundLayerSettings, color_settings: ColorSettings) -> None:
        self._cam_id: int = cam_id
        self._geometry: CenterGeometry = geometry
        self._source: LayerBase = source
        self._timestamp: float | None = None     # the capture time of the frame drawn last
        self._space: SourceSpace = space
        self._premultiply: bool = premultiply
        self._mask_texture: Texture = mask_texture

        # Configuration
        self.settings: CenterForegroundLayerSettings = settings

        # FBOs
        self._roi_fbo: Fbo = Fbo()
        self._premultiplied_fbo: Fbo = Fbo()    # the cut-out times the mask, with `premultiply`
        self._look: ForegroundLook = ForegroundLook(cam_id, settings, color_settings)

        # Shaders
        self._roi_shader = DrawRoi()
        self._premultiply_shader = MaskMultiply()

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._look.texture

    @property
    def timestamp(self) -> float | None:
        """The capture time of the camera frame the foreground it holds was made from, its source's; None without one."""
        return self._timestamp

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._roi_fbo.allocate(width, height, internal_format)
        if self._premultiply:
            self._premultiplied_fbo.allocate(width, height, internal_format)
            self._premultiply_shader.allocate()
        self._look.allocate(width, height, internal_format)
        self._roi_shader.allocate()

    def deallocate(self) -> None:
        self._roi_fbo.deallocate()
        self._premultiplied_fbo.deallocate()
        self._premultiply_shader.deallocate()
        self._look.deallocate()
        self._roi_shader.deallocate()

    def update(self) -> None:
        """Render the foreground centered by CenterGeometry in its space, with the foreground's look."""
        if self._geometry.lost:
            self._look.clear()
            self._timestamp = None

        source: Texture = self._source.texture
        if not self._geometry.ready or not source.allocated:
            return
        self._timestamp = self._source.timestamp if isinstance(self._source, HasCaptureTime) else None

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Render foreground with its ROI
        self._roi_fbo.begin()
        self._roi_shader.use(source, *self._geometry.source_roi(self._space, source))
        self._roi_fbo.end()

        # A whole image: the player's colors on black, as a masked source arrives
        cut: Texture = self._roi_fbo
        if self._premultiply and self._mask_texture.allocated:
            self._premultiplied_fbo.begin()
            self._premultiply_shader.use(self._roi_fbo, self._mask_texture)
            self._premultiplied_fbo.end()
            cut = self._premultiplied_fbo

        Style.pop_style()
        self._look.process(cut, self._mask_texture)
