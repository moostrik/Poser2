"""LayerCompositor - Composites multiple layer textures with optional LUT color grading."""

from __future__ import annotations

from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_RGBA16F, glClear, glClearColor

from modules.gl import Fbo, Texture, BlendState
from ..layer_base import LayerBase
from modules.settings import Field, BaseSettings, Group
from .lut_grade import LutGrade, LutSettings


# ============================================================================
# Configuration
# ============================================================================

class LayerCompositorSettings(BaseSettings):
    """Configuration for LayerCompositor."""
    blend_mode: Field[BlendState.BlendMode] = Field(BlendState.BlendMode.ALPHA)
    lut:        Group[LutSettings]     = Group(LutSettings)


# ============================================================================
# Layer Implementation
# ============================================================================

class LayerCompositor(LayerBase):
    """Composites multiple layer textures with optional LUT color grading.

    Takes a list of layers, renders their textures to an internal FBO using
    the configured blend mode, then optionally applies a 3D LUT for color grading.

    Usage:
        layers = [fluid_layer, ms_mask_layer]
        composite = LayerCompositor(layers, settings)
        composite.allocate(width, height, GL_RGBA16F)

        # Each frame:
        composite.update()  # Composites layers
        composite.draw()    # Draws with LUT applied
    """

    def __init__(
        self,
        layers: list[LayerBase],
        settings: LayerCompositorSettings | None = None
    ) -> None:
        """Initialize LayerCompositor.

        Args:
            layers: List of layers to composite (uses their .texture property)
            settings: Layer configuration
        """
        self._layers: list[LayerBase] = layers
        self.settings: LayerCompositorSettings = settings or LayerCompositorSettings()

        # Composition FBO (before LUT)
        self._composite_fbo: Fbo = Fbo()

        # The LUT the composite is drawn through
        self._grade: LutGrade = LutGrade(self.settings.lut)

    @property
    def texture(self) -> Texture:
        """Output texture (composited, before LUT - for chaining)."""
        return self._composite_fbo

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        """Allocate resources."""
        w = width or 1920
        h = height or 1080
        fmt = internal_format or GL_RGBA16F

        self._composite_fbo.allocate(w, h, fmt)
        self._grade.allocate()

    def deallocate(self) -> None:
        """Clean up resources."""
        self._composite_fbo.deallocate()
        self._grade.deallocate()

    def compose(self, entries: list[tuple[Texture, float]]) -> None:
        """Composite textures with per-entry opacity to internal FBO.

        Args:
            entries: List of (texture, opacity) pairs to composite.
        """
        if not self._composite_fbo.allocated:
            return

        # Begin rendering to composite FBO
        self._composite_fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        BlendState.push()
        BlendState.set_blend_mode(self.settings.blend_mode)

        for tex, opacity in entries:
            if tex.allocated and opacity > 0:
                tex.draw(opacity=opacity)

        self._composite_fbo.end()
        BlendState.pop()

    def set_layers(self, layers: list[LayerBase]) -> None:
        self._layers = layers

    def update(self) -> None:
        """Composite all layer textures to internal FBO."""
        self.compose([(layer.texture, 1.0) for layer in self._layers
                      if hasattr(layer, 'texture') and layer.texture.allocated])

    def draw(self) -> None:
        """Draw the composited result with LUT applied."""
        if not self._composite_fbo.allocated:
            return
        self._grade.draw(self._composite_fbo)
