"""Auto-detecting visualization field wrapper.

Automatically chooses visualization method based on texture format.
Ported from ofxFlowTools ftVisualizationField.h
"""

from modules.gl import Texture

from .. import FlowUtil
from .field_base import VisualizationFieldSettings
from .signed_field import SignedField
from .velocity_field import VelocityField

from modules.utils import HotReloadMethods


class Visualizer:
    """Auto-detecting visualization wrapper for flow textures.

    Automatically chooses rendering method based on texture format:
    - RGB/RGBA (3-4 channels): Direct draw
    - RG (2 channels): Velocity field visualization
    - R (1 channel): Direct draw (or temperature field in future)

    For RG textures, access velocity_field.settings.toggle_scalar to switch modes.
    RG textures are always visualized since raw output can contain negatives.
    """

    def __init__(self, settings: VisualizationFieldSettings | None = None) -> None:
        self.settings: VisualizationFieldSettings = settings or VisualizationFieldSettings()
        self.settings.bind_all(lambda v: self._on_settings_changed())

        # Velocity visualization for 2-channel data
        self.velocity_field: VelocityField = VelocityField(self.settings)
        # Two-color visualization for signed scalars
        self.signed_field: SignedField = SignedField(self.settings)

        self._allocated: bool = False
        self._current_texture: Texture | None = None
        self._use_velocity_field: bool = False
        self._use_signed_field: bool = False

        # hot_reload = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int) -> None:
        """Allocate visualization resources.

        Args:
            width: Visualization width
            height: Visualization height
        """
        self.velocity_field.allocate(width, height)
        self.signed_field.allocate(width, height)
        self._allocated = True

    def deallocate(self) -> None:
        """Deallocate all resources."""
        self.velocity_field.deallocate()
        self.signed_field.deallocate()
        self._allocated = False
        self._current_texture = None

    @property
    def texture(self) -> Texture:
        """Output texture (velocity or signed field FBO, or passthrough)."""
        if self._use_signed_field:
            return self.signed_field.texture
        elif self._use_velocity_field:
            return self.velocity_field.texture
        elif self._current_texture is not None:
            return self._current_texture
        else:
            raise RuntimeError("Visualizer: no texture available, call process() first")

    def process(self, texture: Texture, vector: bool = False) -> None:
        """Process texture for visualization (FBO operations happen here).

        `vector` draws it as a velocity whatever its format: with three or more channels, a 3D one.
        """
        self._use_signed_field = False
        if not self._allocated or not texture.allocated:
            self._current_texture = None
            self._use_velocity_field = False
            return

        self._current_texture = texture

        # Get texture format
        num_channels = FlowUtil.get_num_channels(texture.internal_format)

        # Visualize 2-channel velocity data (RG32F)
        # Always visualized since raw output can contain negatives
        if num_channels <= 2 or vector:
            self._use_velocity_field = True
            # Sync scale with velocity visualization
            self.velocity_field.settings.scale = self.settings.scale

            # Render visualization to FBO
            self.velocity_field.set(texture)
            self.velocity_field.update()
        else:
            # RGB/RGBA (3-4 channels): passthrough, no FBO needed
            self._use_velocity_field = False

    def process_signed(self, texture: Texture, depth: bool = False, channel: int = 0) -> None:
        """Process a scalar texture for the two-color visualization: its `channel`; `depth` is red toward the camera, green away."""
        if not self._allocated or not texture.allocated:
            self._current_texture = None
            self._use_velocity_field = False
            self._use_signed_field = False
            return

        self._current_texture = texture
        self._use_velocity_field = False
        self._use_signed_field = True
        self.signed_field.set(texture, depth, channel)
        self.signed_field.update()

    def _on_settings_changed(self) -> None:
        """Handle settings changes."""
        pass  # Settings changes are applied during process()
