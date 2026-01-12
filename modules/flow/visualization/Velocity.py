"""Flow Visualization Layer.

Renders velocity fields with color encoding or arrow visualization.
Ported from ofxFlowTools ftVisualizationField.h
"""
from dataclasses import dataclass, field
from enum import Enum

from OpenGL.GL import *  # type: ignore

from modules.gl import Fbo, Texture

from .. import FlowBase, FlowConfigBase
from .shaders import VelocityDirectionMap, VelocityArrowField


class VisualizationMode(Enum):
    """Velocity visualization rendering modes."""
    DIRECTION_MAP = 0  # HSV color encoding (direction->hue, magnitude->saturation)
    ARROW_FIELD = 1    # Procedural arrow field


@dataclass
class VelocityConfig(FlowConfigBase):
    scale: float = field(
        default=1.0,
        metadata={"min": 0.0, "max": 10.0, "label": "Scale", "description": "Velocity magnitude scale"}
    )
    mode: VisualizationMode = field(
        default=VisualizationMode.DIRECTION_MAP,
        metadata={"label": "Mode", "description": "Visualization rendering mode"}
    )
    spacing: float = field(
        default=8.0,
        metadata={"min": 4.0, "max": 16.0, "label": "Grid Spacing", "description": "Distance between arrows (pixels)"}
    )
    arrow_length: float = field(
        default=8.0,
        metadata={"min": 0.5, "max": 64.0, "label": "Arrow Length", "description": "Arrow length in pixels"}
    )
    arrow_thickness: float = field(
        default=0.8,
        metadata={"min": 0.5, "max": 2.5, "label": "Arrow Thickness", "description": "Arrow line thickness"}
    )


class Velocity(FlowBase):
    """Visualize velocity/flow fields using color encoding or arrows.

    Supports two visualization modes:
    - DIRECTION_MAP: HSV color encoding (direction->hue, magnitude->saturation)
    - ARROW_FIELD: Procedural arrow field (grid-based sampling)
    """

    def __init__(self, config: VelocityConfig | None = None) -> None:
        super().__init__()

        # Define internal formats
        self.input_internal_format = GL_RG32F   # Velocity input
        self.output_internal_format = GL_RGBA8  # Display output

        # Configuration with change notification
        self.config: VelocityConfig = config or VelocityConfig()
        self.config.add_listener(self._on_config_changed)

        self._velocity_texture: Texture | None = None

        # Shaders
        self._direction_shader: VelocityDirectionMap = VelocityDirectionMap()
        self._arrow_shader: VelocityArrowField = VelocityArrowField()

        self._needs_update: bool = True

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate visualization layer."""
        # Allocate shaders
        self._direction_shader.allocate()
        self._arrow_shader.allocate()
        # Call base allocate for input/output FBOs
        super().allocate(width, height, output_width, output_height)

    def deallocate(self) -> None:
        """Release resources."""
        self._direction_shader.deallocate()
        self._arrow_shader.deallocate()
        super().deallocate()

    def update(self) -> None:
        """Render velocity visualization to internal FBO."""
        if not self._allocated or self._velocity_texture is None:
            return

        if not self._needs_update:
            return

        self._needs_update = False

        self.output_fbo.clear(0.0, 0.0, 0.0, 0.0)

        if self.config.mode == VisualizationMode.DIRECTION_MAP:
            self._direction_shader.reload()
            self._direction_shader.use(
                self.output_fbo,
                self._velocity_texture,
                self.config.scale
            )
        else:  # ARROW_FIELD
            self._arrow_shader.reload()
            self._arrow_shader.use(
                self.output_fbo,
                self._velocity_texture,
                self.config.scale,
                self.config.spacing,
                self.config.arrow_length,
                self.config.arrow_thickness
            )

    def set(self, texture: Texture) -> None:
        """Set the velocity field texture to visualize. """
        # if texture != self._velocity_texture:
        self._velocity_texture = texture
        self._needs_update = True

    def _on_config_changed(self) -> None:
        """Called when config values change."""
        self._needs_update = True
