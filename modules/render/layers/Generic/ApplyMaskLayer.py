"""Applies mask to any source layer (camera, flow, etc.)."""

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.render.layers.LayerBase import LayerBase
from modules.render.layers.centre.CentreMaskLayer import CentreMaskLayer
from modules.render.shaders import MaskApply
from modules.utils.PointsAndRects import Rect

# GL
from modules.gl import Fbo, Texture

from modules.utils.HotReloadMethods import HotReloadMethods


class ApplyMaskLayer(LayerBase):
    """Composites source layer with mask layer using MaskApply shader.

    Takes any source layer (camera, flow, etc.) and applies the mask from
    CentreMaskLayer. Output is the source masked with alpha channel.
    """

    def __init__(self, source_layer: LayerBase, mask_layer: CentreMaskLayer, opacity: float = 1.0) -> None:
        """Initialize mask apply layer.

        Args:
            source_layer: Source layer to be masked (must have texture property)
            mask_layer: CentreMaskLayer providing the mask
            opacity: Overall opacity multiplier [0.0, 1.0]
        """
        self._source_layer: TextureLayer = source_layer
        self._mask_layer: CentreMaskLayer = mask_layer
        self._fbo: Fbo = Fbo()
        self._shader: MaskApply = MaskApply()

        self.opacity: float = opacity

        HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        """Output texture for external use."""
        return self._fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        """Apply mask to source layer."""
        glDisable(GL_BLEND)
        glColor4f(1.0, 1.0, 1.0, 1.0)

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Get source and mask textures
        source_tex = self._source_layer.texture
        mask_tex = self._mask_layer.texture

        # Apply mask shader
        self._shader.use(
            self._fbo.fbo_id,
            source_tex.tex_id,
            mask_tex.tex_id,
            self.opacity
        )

        glEnable(GL_BLEND)
