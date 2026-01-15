"""Flow Visualization Layer.

Renders velocity fields with color encoding or arrow visualization.
Ported from ofxFlowTools ftVisualizationField.h
"""
from dataclasses import dataclass, field
from enum import Enum

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture

from .BaseField import FieldBase, VisualisationFieldConfig
from .shaders import VelocityDirectionMap, VelocityArrowField

from modules.utils.HotReloadMethods import HotReloadMethods



class VelocityField(FieldBase):
    """Visualize velocity/flow fields using color encoding or arrows.

    Supports two visualization modes:
    - Direction Map: HSV color encoding (direction->hue, magnitude->saturation)
    - Arrow Field: Procedural arrow field (grid-based sampling)

    toggle_scalar switches between the two modes.
    scale acts as a general multiplier for both modes.
    """

    def __init__(self, config: VisualisationFieldConfig | None = None) -> None:
        super().__init__()

        # Configuration with change notification
        self.config: VisualisationFieldConfig = config or VisualisationFieldConfig()
        self.config.add_listener(self._on_config_changed)

        self._velocity_texture: Texture | None = None

        # Shaders
        self._direction_shader: VelocityDirectionMap = VelocityDirectionMap()
        self._arrow_shader: VelocityArrowField = VelocityArrowField()

        self._needs_update: bool = True

        hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int) -> None:
        """Allocate visualization layer."""
        super().allocate(width, height)
        self._direction_shader.allocate()
        self._arrow_shader.allocate()

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

        self._fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fbo.begin()
        if self.config.toggle_scalar:
            # Direction map mode
            self._direction_shader.reload()
            self._direction_shader.use(
                self._velocity_texture,
                self.config.scale
            )
        else:
            # Arrow field mode
            self._arrow_shader.reload()
            self._arrow_shader.use(
                self._velocity_texture,
                self.config.scale,
                self.config.spacing,
                self.config.element_length,
                self.config.element_width
            )
        self._fbo.end()

    def set(self, texture: Texture) -> None:
        """Set the velocity field texture to visualize."""
        self._velocity_texture = texture
        self._needs_update = True

    def _on_config_changed(self) -> None:
        """Called when config values change."""
        self._needs_update = True
