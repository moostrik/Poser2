"""Auto-detecting visualization field wrapper.

Automatically chooses visualization method based on texture format.
Ported from ofxFlowTools ftVisualizationField.h
"""
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Texture
from modules.utils.PointsAndRects import Rect

from .. import FlowConfigBase, FlowUtil
from .BaseField import VisualisationFieldConfig
from .VelocityField import VelocityField

from modules.utils.HotReloadMethods import HotReloadMethods


class Visualizer:
    """Auto-detecting visualization wrapper for flow textures.

    Automatically chooses rendering method based on texture format:
    - RGB/RGBA (3-4 channels): Direct draw
    - RG (2 channels): Velocity field visualization
    - R (1 channel): Direct draw (or temperature field in future)

    For RG textures, access velocity_field.config.toggle_scalar to switch modes.
    RG textures are always visualized since raw output can contain negatives.
    """

    def __init__(self, config: VisualisationFieldConfig | None = None) -> None:
        self._config: VisualisationFieldConfig = config or VisualisationFieldConfig()
        self._config.add_listener(self._on_config_changed)

        # Velocity visualization for 2-channel data
        self.velocity_field: VelocityField = VelocityField(self._config)

        self._allocated: bool = False

        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def config(self) -> VisualisationFieldConfig:
        """Get visualization configuration."""
        return self._config

    def allocate(self, width: int, height: int) -> None:
        """Allocate visualization resources.

        Args:
            width: Visualization width
            height: Visualization height
        """
        self.velocity_field.allocate(width, height)
        self._allocated = True

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self.velocity_field.deallocate()
        self._allocated = False

    def draw(self, texture: Texture, rect: Rect) -> None:
        """Auto-detect format and draw appropriately.

        Args:
            texture: Source texture to visualize
            rect: Draw rectangle
        """
        if not self._allocated or not texture.allocated:
            return

        # Get texture format
        num_channels = FlowUtil.get_num_channels(texture.internal_format)  # type: ignore

        # Visualize 2-channel velocity data (RG32F)
        # Always visualized since raw output can contain negatives
        # Use velocity_field.config.toggle_scalar to switch modes
        if num_channels <= 2:
            # Sync scale with velocity visualization
            self.velocity_field.config.scale = self._config.scale

            # Render visualization
            self.velocity_field.set(texture)
            self.velocity_field.update()
            self.velocity_field.draw(rect)

        # Direct draw for RGB/RGBA (3-4 channels) or R (1 channel)
        else:
            texture.draw(rect.x, rect.y, rect.width, rect.height)

    def _on_config_changed(self) -> None:
        """Handle config changes."""
        pass  # Config changes are applied during draw()
