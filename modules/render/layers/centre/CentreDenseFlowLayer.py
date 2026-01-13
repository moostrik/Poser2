"""Renders centered and rotated optical flow visualization with optional mask."""

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreGeometry import CentreGeometry
from modules.render.shaders import DrawRoi, MaskApply
from modules.gl import Fbo, Texture


class CentreDenseFlowLayer(LayerBase):
    """Renders optical flow visualization cropped and rotated around pose anchor points.

    Reads anchor geometry from CentreGeometry and applies DrawRoi shader to flow visualization.
    Optionally applies mask texture for compositing.
    """

    def __init__(self, anchor_calc: CentreGeometry, flow_texture: Texture, mask_texture: Texture | None = None, mask_opacity: float = 1.0) -> None:
        self._anchor_calc: CentreGeometry = anchor_calc
        self._flow_texture: Texture = flow_texture
        self._mask_texture: Texture | None = mask_texture

        # FBOs
        self._flow_fbo: Fbo = Fbo()
        self._masked_fbo: Fbo = Fbo()
        self._output_fbo: Fbo = self._masked_fbo if self._mask_texture else self._flow_fbo

        # Shaders
        self._roi_shader = DrawRoi()
        self._mask_shader = MaskApply()

        self.mask_opacity: float = mask_opacity
        self.use_mask: bool = True  # Toggle for mask application

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._output_fbo

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
        # Disable blending during FBO rendering
        glDisable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self._flow_fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Check if valid anchor data exists
        if not self._anchor_calc.has_pose:
            if self._mask_texture and self.use_mask:
                self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            glEnable(GL_BLEND)
            return

        # Render flow with ROI from anchor calculator (bbox-space geometry, like mask)
        self._flow_fbo.begin()
        self._roi_shader.use(
            self._flow_texture,
            self._anchor_calc.bbox_crop_roi,
            self._anchor_calc.bbox_rotation,
            self._anchor_calc.bbox_rotation_center,
            self._anchor_calc.bbox_aspect,
            False,
            False
        )
        self._flow_fbo.end()

        # Apply mask if provided and enabled
        if self._mask_texture and self.use_mask:
            self._masked_fbo.clear(0.0, 0.0, 0.0, 0.0)
            self._masked_fbo.begin()
            self._mask_shader.use(
                self._flow_fbo,
                self._mask_texture,
                self.mask_opacity
            )
            self._masked_fbo.end()
            self._output_fbo = self._masked_fbo
        else:
            self._output_fbo = self._flow_fbo

        glEnable(GL_BLEND)