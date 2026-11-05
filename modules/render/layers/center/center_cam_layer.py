"""Renders centered and rotated camera view based on pose anchor points with optional mask."""

# Local application imports
from modules.settings import Field, BaseSettings
from ..layer_base import LayerBase, HasCaptureTime
from .center_geometry import CenterGeometry
from ...shaders import Blend, DrawRoi, MaskApply

# GL
from modules.gl import Fbo, SwapFbo, Texture


class CenterCamLayerSettings(BaseSettings):
    """Configuration for CenterCamLayer camera rendering."""
    blend_factor:   Field[float] = Field(0.2, min=0.0, max=1.0, description="Camera frame temporal blending")
    mask_opacity:   Field[float] = Field(1.0, min=0.0, max=1.0, description="Mask alpha strength")
    use_mask:       Field[bool]  = Field(True, description="Apply mask to camera output")

class CenterCamLayer(LayerBase):
    """Renders camera image cropped and rotated around pose anchor points.

    Reads anchor geometry from CenterGeometry and applies DrawRoi shader
    followed by temporal blending. Optionally applies mask texture for compositing.
    `source` is the layer whose texture it draws; `timestamp` is that frame's capture time, forwarded.
    """

    def __init__(self, cam_id: int, geometry: CenterGeometry, source: LayerBase, mask_texture: Texture, settings: CenterCamLayerSettings) -> None:
        self._cam_id: int = cam_id
        self._geometry: CenterGeometry = geometry
        self._source: LayerBase = source
        self._mask_texture: Texture = mask_texture
        self._timestamp: float | None = None     # the capture time of the frame drawn last

        # Configuration
        self.settings: CenterCamLayerSettings = settings or CenterCamLayerSettings()

        # FBOs
        self._cam_fbo: Fbo = Fbo()
        self._cam_blend_fbo: SwapFbo = SwapFbo()
        self._masked_fbo: Fbo = Fbo()
        self._output_fbo: Fbo | SwapFbo = self._masked_fbo if self._mask_texture else self._cam_blend_fbo

        # Shaders
        self._roi_shader = DrawRoi()
        self._blend_shader = Blend()
        self._mask_shader = MaskApply()

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._output_fbo        # the SwapFbo itself: always its newest image, safe to keep

    @property
    def timestamp(self) -> float | None:
        """The capture time of the camera frame the view holds, its source's; None without one."""
        return self._timestamp

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._cam_fbo.allocate(width, height, internal_format)
        self._cam_blend_fbo.allocate(width, height, internal_format)
        if self._mask_texture:
            self._masked_fbo.allocate(width, height, internal_format)
        self._roi_shader.allocate()
        self._blend_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._cam_fbo.deallocate()
        self._cam_blend_fbo.deallocate()
        self._masked_fbo.deallocate()
        self._roi_shader.deallocate()
        self._blend_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        """Render camera crop using anchor geometry, optionally with mask."""
        if self._geometry.lost:
            self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._cam_blend_fbo.swap()
            self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            if self._mask_texture and self.settings.use_mask:
                self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._timestamp = None
            return

        source: Texture = self._source.texture
        if not self._geometry.ready or not source.allocated:
            return
        self._timestamp = self._source.timestamp if isinstance(self._source, HasCaptureTime) else None

        # Render camera image with ROI from anchor calculator. Cleared first: a source with
        # transparent pixels (depth) must not leave the last frame showing through them.
        cam_aspect: float = source.width / source.height
        self._cam_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._cam_fbo.begin()
        self._roi_shader.use(
            source,
            self._geometry.image_geometry.crop_roi,
            self._geometry.image_geometry.rotation,
            self._geometry.image_geometry.rotation_center,
            cam_aspect,
        )
        self._cam_fbo.end()

        # Temporal blending
        self._cam_blend_fbo.swap()
        self._cam_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._cam_blend_fbo.begin()
        self._blend_shader.use(
            self._cam_blend_fbo.back_texture,
            self._cam_fbo.texture,
            self.settings.blend_factor
        )
        self._cam_blend_fbo.end()

        # Apply mask if provided and enabled
        if self.settings.use_mask:
            self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._masked_fbo.begin()
            self._mask_shader.use(
                self._cam_blend_fbo.texture,
                self._mask_texture,
                self.settings.mask_opacity
            )
            self._masked_fbo.end()
            self._output_fbo = self._masked_fbo
        else:
            self._output_fbo = self._cam_blend_fbo
