"""Signed scalar field visualization."""

from modules.gl import Texture

from .field_base import FieldBase, VisualizationFieldSettings
from .shaders import SignedMap


class SignedField(FieldBase):
    """Visualize a scalar field in two colors, one per sign; settings.scalar_scale multiplies the value."""

    def __init__(self, settings: VisualizationFieldSettings | None = None) -> None:
        super().__init__()

        self.settings: VisualizationFieldSettings = settings or VisualizationFieldSettings()

        self._field_texture: Texture | None = None
        self._depth: bool = False
        self._channel: int = 0
        self._unit: float = 1.0

        self._shader: SignedMap = SignedMap()

    def allocate(self, width: int, height: int) -> None:
        """Allocate visualization layer."""
        super().allocate(width, height)
        self._shader.allocate()

    def deallocate(self) -> None:
        """Release resources."""
        self._shader.deallocate()
        super().deallocate()

    def set(self, texture: Texture, depth: bool = False, channel: int = 0, unit: float = 1.0) -> None:
        """Set the scalar field to visualize, in `channel`; `depth` draws it red toward the camera and green away.
        `unit` turns the field's values into the unit shown, so 1 of it is full color at scalar_scale 1."""
        self._field_texture = texture
        self._depth = depth
        self._channel = channel
        self._unit = unit

    def update(self) -> None:
        """Render the field to the internal FBO."""
        if not self._allocated or self._field_texture is None:
            return

        self._shader.reload()
        self._fbo.begin()
        self._shader.use(self._field_texture, self.settings.scalar_scale * self._unit, self._depth, self._channel)
        self._fbo.end()
