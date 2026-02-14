"""Renders centered and rotated foreground with temporal blending and optional mask."""

# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import DrawRoi, Blend, MaskApply, CelShade
from modules.gl import Fbo, SwapFbo, Texture, Blit
from modules.gl.shaders import Sharpen
from modules.gl import Style

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class CentreFrgConfig(ConfigBase):
    """Configuration for CentreFrgLayer foreground rendering."""
    blend_factor: float = config_field(0.2, min=0.0, max=1.0, description="Foreground temporal blending")
    exposure: float = config_field(1.0, min=0.1, max=3.0, description="Exposure multiplier")
    gamma: float = config_field(1.0, min=0.5, max=2.0, description="Gamma correction")
    offset: float = config_field(0.0, min=-0.5, max=0.5, description="Exposure offset")
    contrast: float = config_field(1.0, min=0.5, max=2.0, description="Contrast")
    saturation: float = config_field(1.2, min=0.5, max=2.0, description="Color saturation")
    # Cel shade
    levels: int = config_field(4, min=2, max=8, description="Number of color bands")
    smoothness: float = config_field(0.1, min=0.0, max=0.5, description="Gradient between bands")
    # Post
    sharpen: float = config_field(0.5, min=0.0, max=2.0, description="Sharpen result (0.0 = off)")
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
        self._roi_fbo: Fbo = Fbo()
        self._blend_fbo: SwapFbo = SwapFbo()
        self._effect_fbo: SwapFbo = SwapFbo()

        # Shaders
        self._roi_shader = DrawRoi()
        self._blend_shader = Blend()
        self._cel_shade_shader = CelShade()
        self._sharpen_shader = Sharpen()
        self._mask_shader = MaskApply()
        self.hot_reload = HotReloadMethods(self.__class__)

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._effect_fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._roi_fbo.allocate(width, height, internal_format)
        self._blend_fbo.allocate(width, height, internal_format)
        self._effect_fbo.allocate(width, height, internal_format)

        self._roi_shader.allocate()
        self._blend_shader.allocate()
        self._mask_shader.allocate()
        self._cel_shade_shader.allocate()
        self._sharpen_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._roi_fbo.deallocate()
        self._blend_fbo.deallocate()
        self._effect_fbo.deallocate()

        self._roi_shader.deallocate()
        self._blend_shader.deallocate()
        self._cel_shade_shader.deallocate()
        self._sharpen_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        """Render foreground crop using bbox geometry, with temporal blending and optional mask."""
        if self._geometry.lost:
            self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._blend_fbo.swap()
            self._blend_fbo.clear(0.0, 0.0, 0.0, 0.0)

        if self._geometry.crop_pose_points is None:
            return

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        # Render foreground with bbox ROI
        self._roi_fbo.begin()
        self._roi_shader.use(
            self._frg_texture,
            self._geometry.bbox_geometry.crop_roi,
            self._geometry.bbox_geometry.rotation,
            self._geometry.bbox_geometry.rotation_center,
            self._geometry.bbox_geometry.aspect
        )
        self._roi_fbo.end()

        # Temporal blending
        self._blend_fbo.swap()
        self._blend_fbo.begin()
        self._blend_shader.use(
            self._blend_fbo.back_texture,
            self._roi_fbo.texture,
            self.config.blend_factor
        )
        self._blend_fbo.end()

        self._cel_shade_shader.reload()
        self._effect_fbo.begin()
        self._cel_shade_shader.use(
            self._blend_fbo.texture,
            self.config.exposure, self.config.gamma, self.config.offset,
            self.config.contrast,
            self.config.levels, self.config.smoothness,
            self.config.saturation
        )
        self._effect_fbo.end()

        # Sharpen
        if self.config.sharpen > 0.0:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            self._sharpen_shader.use(self._effect_fbo.back_texture, self.config.sharpen)
            self._effect_fbo.end()

        # Mask
        if self._mask_texture and self.config.use_mask:
            self._effect_fbo.swap()
            self._effect_fbo.begin()
            self._mask_shader.use(self._effect_fbo.back_texture, self._mask_texture)
            self._effect_fbo.end()

        Style.pop_style()