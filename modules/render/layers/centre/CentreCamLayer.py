"""Renders centered and rotated camera view based on pose anchor points with optional mask."""

# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import Blend, DrawRoi, MaskApply

# GL
from modules.gl import Fbo, SwapFbo, Texture, Style, clear_color

from modules.utils import HotReloadMethods


@dataclass
class CentreCamConfig(ConfigBase):
    """Configuration for CentreCamLayer camera rendering."""
    blend_factor: float = config_field(0.5, min=0.0, max=1.0, description="Camera frame temporal blending")
    mask_opacity: float = config_field(1.0, min=0.0, max=1.0, description="Mask alpha strength")
    use_mask: bool = config_field(True, description="Apply mask to camera output")

class CentreCamLayer(LayerBase):
    """Renders camera image cropped and rotated around pose anchor points.

    Reads anchor geometry from CentreGeometry and applies DrawRoi shader
    followed by temporal blending. Optionally applies mask texture for compositing.
    """

    def __init__(self, geometry: CentreGeometry, cam_texture: Texture, mask_texture: Texture | None = None, config: CentreCamConfig | None = None) -> None:
        self._geometry: CentreGeometry = geometry
        self._cam_texture: Texture = cam_texture
        self._mask_texture: Texture | None = mask_texture

        # Configuration
        self.config: CentreCamConfig = config or CentreCamConfig()

        # FBOs
        self._cam_fbo: Fbo = Fbo()
        self._cam_blend_fbo: SwapFbo = SwapFbo()
        self._masked_fbo: Fbo = Fbo()
        self._output_fbo: Fbo | SwapFbo = self._masked_fbo if self._mask_texture else self._cam_blend_fbo

        # Shaders
        self._roi_shader = DrawRoi()
        self._blend_shader = Blend()
        self._mask_shader = MaskApply()

        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._output_fbo.texture

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
            if self._mask_texture and self.config.use_mask:
                self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            return

        if self._geometry.crop_pose_points is None:
            return

        # Render camera image with ROI from anchor calculator
        cam_aspect: float = self._cam_texture.width / self._cam_texture.height
        self._cam_fbo.begin()
        self._roi_shader.use(
            self._cam_texture,
            self._geometry.image_geometry.crop_roi,
            self._geometry.image_geometry.rotation,
            self._geometry.image_geometry.rotation_center,
            cam_aspect,
        )
        self._cam_fbo.end()

        # Temporal blending
        self._cam_blend_fbo.swap()
        self._cam_blend_fbo.begin()
        self._blend_shader.use(
            self._cam_blend_fbo.back_texture,
            self._cam_fbo.texture,
            self.config.blend_factor
        )
        self._cam_blend_fbo.end()

        # Apply mask if provided and enabled
        if self._mask_texture and self.config.use_mask:
            self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._masked_fbo.begin()
            self._mask_shader.use(
                self._cam_blend_fbo.texture,
                self._mask_texture,
                self.config.mask_opacity
            )
            self._masked_fbo.end()
            self._output_fbo = self._masked_fbo
        else:
            self._output_fbo = self._cam_blend_fbo
