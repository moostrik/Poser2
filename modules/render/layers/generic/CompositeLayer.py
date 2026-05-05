"""CompositeLayer - Composites multiple layer textures with optional LUT color grading."""

from __future__ import annotations
import logging
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from modules.gl import Fbo, Texture, Style
from modules.gl.shaders import Blit, Lut
from ..LayerBase import LayerBase
from modules.settings import Field, BaseSettings, Widget, Group

from modules.utils import HotReloadMethods

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class LutSettings(BaseSettings):
    """LUT color grading configuration."""
    folder:   Field[str]       = Field("data/luts", access=Field.INIT, description="LUT folder")
    files:    Field[list[str]] = Field([""], visible=False)
    file:     Field[str]       = Field("", widget=Widget.text_select, options=files, description="LUT file")
    rescan:   Field[bool]      = Field(False, widget=Widget.button, description="Rescan LUTs")
    strength: Field[float]     = Field(1.0, min=0.0, max=1.0)


class CompositeLayerSettings(BaseSettings):
    """Configuration for CompositeLayer."""
    blend_mode: Field[Style.BlendMode] = Field(Style.BlendMode.ALPHA)
    lut:        Group[LutSettings]     = Group(LutSettings)


# ============================================================================
# Layer Implementation
# ============================================================================

class CompositeLayer(LayerBase):
    """Composites multiple layer textures with optional LUT color grading.

    Takes a list of layers, renders their textures to an internal FBO using
    the configured blend mode, then optionally applies a 3D LUT for color grading.

    Usage:
        layers = [fluid_layer, ms_mask_layer]
        composite = CompositeLayer(layers, config)
        composite.allocate(width, height, GL_RGBA16F)

        # Each frame:
        composite.update()  # Composites layers
        composite.draw()    # Draws with LUT applied
    """

    def __init__(
        self,
        layers: list[LayerBase],
        config: CompositeLayerSettings | None = None
    ) -> None:
        """Initialize CompositeLayer.

        Args:
            layers: List of layers to composite (uses their .texture property)
            config: Layer configuration
        """
        self._layers: list[LayerBase] = layers
        self.config: CompositeLayerSettings = config or CompositeLayerSettings()

        # Composition FBO (before LUT)
        self._composite_fbo: Fbo = Fbo()

        # LUT shader
        self._lut_shader: Lut = Lut()
        self._current_lut: str = ""

        # Blit shader for passthrough when no LUT
        self._blit: Blit = Blit()

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

        # Scan LUTs now and whenever folder changes or rescan is pressed
        self._scan_luts()
        self.config.lut.bind(LutSettings.folder, lambda _: self._scan_luts())
        self.config.lut.bind(LutSettings.rescan, lambda v: self._scan_luts() if v else None)

    @property
    def texture(self) -> Texture:
        """Output texture (composited, before LUT - for chaining)."""
        return self._composite_fbo.texture

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        """Allocate resources."""
        w = width or 1920
        h = height or 1080
        fmt = internal_format or GL_RGBA16F

        self._composite_fbo.allocate(w, h, fmt)
        self._lut_shader.allocate()
        self._blit.allocate()

        # Load initial LUT if configured
        self._load_lut_if_changed()

    def deallocate(self) -> None:
        """Clean up resources."""
        self._composite_fbo.deallocate()
        self._lut_shader.deallocate()
        self._blit.deallocate()

    def _scan_luts(self) -> None:
        """Scan lut folder for .cube files and update files list. Resets selection if no longer valid."""
        lut_dir = Path(self.config.lut.folder)
        if lut_dir.exists():
            filenames = [f.name for f in sorted(lut_dir.glob("*.cube"))]
        else:
            logger.warning("LUT directory not found: %s", lut_dir)
            filenames = []
        self.config.lut.files = filenames
        if self.config.lut.file and self.config.lut.file not in filenames:
            self.config.lut.file = ""

    def _load_lut_if_changed(self) -> None:
        """Load LUT file if selection changed."""
        if self.config.lut.file != self._current_lut:
            self._current_lut = self.config.lut.file
            if self.config.lut.file:
                lut_path = Path(self.config.lut.folder) / self.config.lut.file
                self._lut_shader.load_cube(str(lut_path))

    def compose(self, entries: list[tuple[Texture, float]]) -> None:
        """Composite textures with per-entry opacity to internal FBO.

        Args:
            entries: List of (texture, opacity) pairs to composite.
        """
        if not self._composite_fbo.allocated:
            return

        # Check for LUT changes
        self._load_lut_if_changed()

        # Begin rendering to composite FBO
        self._composite_fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        Style.push_style()
        Style.set_blend_mode(self.config.blend_mode)

        for tex, opacity in entries:
            if tex.allocated and opacity > 0:
                self._blit.use(tex, opacity)

        self._composite_fbo.end()
        Style.pop_style()

    def update(self) -> None:
        """Composite all layer textures to internal FBO."""
        self.compose([(layer.texture, 1.0) for layer in self._layers
                      if hasattr(layer, 'texture') and layer.texture.allocated])

    def draw(self) -> None:
        """Draw the composited result with LUT applied."""
        if not self._composite_fbo.allocated:
            return

        # Apply LUT if loaded, otherwise just blit
        if self._lut_shader.lut_loaded and self.config.lut.file and self.config.lut.strength > 0:
            self._lut_shader.use(self._composite_fbo.texture, self.config.lut.strength)
        else:
            self._blit.use(self._composite_fbo.texture)
