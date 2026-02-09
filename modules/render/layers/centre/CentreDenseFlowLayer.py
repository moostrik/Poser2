"""Renders centered and rotated optical flow visualization with optional mask."""

# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.ConfigBase import ConfigBase, config_field
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import DrawRoi, MaskApply
from modules.gl import Fbo, Texture, Style


@dataclass
class CentreDlowConfig(ConfigBase):
    """Configuration for CentreDenseFlowLayer optical flow visualization."""
    mask_opacity: float = config_field(1.0, min=0.0, max=1.0, description="Flow mask opacity")
    use_mask: bool = config_field(True, description="Enable flow masking")


class CentreDenseFlowLayer(LayerBase):
    """Renders optical flow visualization cropped and rotated around pose anchor points.

    Reads anchor geometry from CentreGeometry and applies DrawRoi shader to flow visualization.
    Optionally applies mask texture for compositing.
    """

    def __init__(self, geometry: CentreGeometry, flow_texture: Texture, mask_texture: Texture | None = None, config: CentreDlowConfig | None = None) -> None:
        self._geometry: CentreGeometry = geometry
        self._flow_texture: Texture = flow_texture
        self._mask_texture: Texture | None = mask_texture

        # Configuration
        self.config: CentreDlowConfig = config or CentreDlowConfig()

        # FBOs
        self._flow_fbo: Fbo = Fbo()
        self._masked_fbo: Fbo = Fbo()
        self._output_fbo: Fbo = self._masked_fbo if self._mask_texture else self._flow_fbo

        # Shaders
        self._roi_shader = DrawRoi()
        self._mask_shader = MaskApply()

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._output_fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._flow_fbo.allocate(width, height, internal_format)
        if self._mask_texture:
            self._masked_fbo.allocate(width, height, internal_format)
        self._roi_shader.allocate()
        self._mask_shader.allocate()

    def deallocate(self) -> None:
        self._flow_fbo.deallocate()
        self._masked_fbo.deallocate()
        self._roi_shader.deallocate()
        self._mask_shader.deallocate()

    def update(self) -> None:
        """Render flow crop using anchor geometry, optionally with mask."""
        if self._geometry.lost:
            self._flow_fbo.clear()
            if self._mask_texture and self.config.use_mask:
                self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            return

        # Check if valid geometry exists
        if self._geometry.crop_pose_points is None:
            return

        # Render flow with ROI from anchor calculator (bbox-space geometry, like mask)
        self._flow_fbo.begin()
        self._roi_shader.use(
            self._flow_texture,
            self._geometry.bbox_geometry.crop_roi,
            self._geometry.bbox_geometry.rotation,
            self._geometry.bbox_geometry.rotation_center,
            self._geometry.bbox_geometry.aspect
        )
        self._flow_fbo.end()

        # Apply mask if provided and enabled
        if self._mask_texture and self.config.use_mask:
            self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._masked_fbo.begin()
            self._mask_shader.use(
                self._flow_fbo.texture,
                self._mask_texture,
                self.config.mask_opacity
            )
            self._masked_fbo.end()
            self._output_fbo = self._masked_fbo
        else:
            self._output_fbo = self._flow_fbo