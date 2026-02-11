"""CompositeLayer - Composites multiple layer textures with optional LUT color grading."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from OpenGL.GL import *  # type: ignore

from modules.gl import Fbo, Texture, Style
from modules.gl.shaders import Blit, Lut
from modules.render.layers.LayerBase import LayerBase


# ============================================================================
# LUT Selection Enum - Dynamically generated from files/lut/*.cube
# ============================================================================

def _discover_luts() -> type[IntEnum]:
    """Scan files/lut/ for .cube files and create an IntEnum."""
    lut_dir = Path(__file__).parents[4] / "files" / "lut"
    luts = {"NONE": 0}

    if lut_dir.exists():
        for i, cube_file in enumerate(sorted(lut_dir.glob("*.cube")), start=1):
            # Convert filename to enum-safe name: "My LUT" -> "MY_LUT"
            name = cube_file.stem.upper().replace(" ", "_").replace("-", "_")
            luts[name] = i

    return IntEnum("LutSelection", luts)


def _get_lut_path(selection: IntEnum) -> str | None:
    """Get the file path for a LUT selection."""
    if selection.value == 0:  # NONE
        return None

    lut_dir = Path(__file__).parents[4] / "files" / "lut"
    if not lut_dir.exists():
        return None

    # Find matching .cube file by index
    cube_files = sorted(lut_dir.glob("*.cube"))
    idx = selection.value - 1
    if 0 <= idx < len(cube_files):
        return str(cube_files[idx])
    return None


# Generate the enum at module load time
LutSelection = _discover_luts()


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CompositeLayerConfig:
    """Configuration for CompositeLayer."""
    blend_mode: Style.BlendMode = Style.BlendMode.ADDITIVE
    lut: LutSelection = field(default_factory=lambda: LutSelection.NONE)  # type: ignore
    lut_strength: float = 1.0


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
        config: CompositeLayerConfig | None = None
    ) -> None:
        """Initialize CompositeLayer.

        Args:
            layers: List of layers to composite (uses their .texture property)
            config: Layer configuration
        """
        self._layers: list[LayerBase] = layers
        self.config: CompositeLayerConfig = config or CompositeLayerConfig()

        # Composition FBO (before LUT)
        self._composite_fbo: Fbo = Fbo()

        # LUT shader
        self._lut_shader: Lut = Lut()
        self._current_lut: LutSelection = LutSelection.NONE  # type: ignore

        # Blit shader for passthrough when no LUT
        self._blit: Blit = Blit()

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

    def _load_lut_if_changed(self) -> None:
        """Load LUT file if selection changed."""
        if self.config.lut != self._current_lut:
            self._current_lut = self.config.lut
            lut_path = _get_lut_path(self.config.lut)
            if lut_path:
                self._lut_shader.load_cube(lut_path)

    def update(self) -> None:
        """Composite all layer textures to internal FBO."""
        if not self._composite_fbo.allocated:
            return

        # Check for LUT changes
        self._load_lut_if_changed()

        # Begin rendering to composite FBO
        self._composite_fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Set blend mode
        # Style.reset_state()
        Style.push_style()
        Style.set_blend_mode(self.config.blend_mode)

        # Draw all layer textures
        for layer in self._layers:
            try:
                tex = layer.texture
                if tex.allocated:
                    self._blit.use(tex)
            except NotImplementedError:
                # Layer doesn't produce texture output
                continue

        self._composite_fbo.end()
        Style.pop_style()

    def draw(self) -> None:
        """Draw the composited result with LUT applied."""
        if not self._composite_fbo.allocated:
            return

        # Apply LUT if loaded, otherwise just blit
        if self._lut_shader.lut_loaded and self.config.lut_strength > 0:
            self._lut_shader.use(self._composite_fbo.texture, self.config.lut_strength)
        else:
            self._blit.use(self._composite_fbo.texture)
