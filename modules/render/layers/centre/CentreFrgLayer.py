"""Renders centered and rotated foreground with temporal blending and optional mask."""

# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import DrawRoi, Blend, MaskApply
from modules.gl import Fbo, SwapFbo, Texture

from modules.utils import HotReloadMethods


@dataclass
class CentreFrgConfig(ConfigBase):
    """Configuration for CentreFrgLayer foreground rendering."""
    blend_factor: float = config_field(0.2, min=0.0, max=1.0, description="Foreground temporal blending")
    mask_opacity: float = config_field(1.0, min=0.0, max=1.0, description="Foreground mask strength")
    use_mask: bool = config_field(True, description="Apply mask to foreground")


class CentreFrgLayer(LayerBase):
    """Renders foreground image cropped and rotated with temporal blending.

    Uses bbox_geometry for rendering. Optionally applies mask texture for compositing.
    """

    def __init__(self, geometry: CentreGeometry, frg_texture: Texture, mask_texture: Texture | None = None, config: CentreFrgConfig | None = None) -> None:
        self._geometry: CentreGeometry = geometry
        self._frg_texture: Texture = frg_texture
        self._mask_texture: Texture | None = mask_texture

        # Configuration
        self.config: CentreFrgConfig = config or CentreFrgConfig()

        # FBOs
        self._frg_fbo: Fbo = Fbo()
        self._frg_blend_fbo: SwapFbo = SwapFbo()
        self._masked_fbo: Fbo = Fbo()
        self._output_fbo: Fbo | SwapFbo = self._masked_fbo if self._mask_texture else self._frg_blend_fbo

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
        self._frg_fbo.allocate(width, height, internal_format)
        self._frg_blend_fbo.allocate(width, height, internal_format)
        if self._mask_texture:
            self._masked_fbo.allocate(width, height, internal_format)
        self._roi_shader.allocate()
        self._blend_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._frg_fbo.deallocate()
        self._frg_blend_fbo.deallocate()
        self._masked_fbo.deallocate()
        self._roi_shader.deallocate()
        self._blend_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        """Render foreground crop using bbox geometry, with temporal blending and optional mask."""
        if self._geometry.lost:
            self._frg_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._frg_blend_fbo.swap()
            self._frg_blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            if self._mask_texture and self.config.use_mask:
                self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            return

        if self._geometry.crop_pose_points is None:
            return

        # Render foreground with bbox ROI
        self._frg_fbo.begin()
        self._roi_shader.use(
            self._frg_texture,
            self._geometry.bbox_geometry.crop_roi,
            self._geometry.bbox_geometry.rotation,
            self._geometry.bbox_geometry.rotation_center,
            self._geometry.bbox_geometry.aspect
        )
        self._frg_fbo.end()

        # Temporal blending
        self._frg_blend_fbo.swap()
        self._frg_blend_fbo.begin()
        self._blend_shader.use(
            self._frg_blend_fbo.back_texture,
            self._frg_fbo.texture,
            self.config.blend_factor
        )
        self._frg_blend_fbo.end()

        # Apply mask if provided and enabled
        if self._mask_texture and self.config.use_mask:
            self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._masked_fbo.begin()
            self._mask_shader.use(
                self._frg_blend_fbo.texture,
                self._mask_texture,
                self.config.mask_opacity
            )
            self._masked_fbo.end()
            self._output_fbo = self._masked_fbo
        else:
            self._output_fbo = self._frg_blend_fbo