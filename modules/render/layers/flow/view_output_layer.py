"""The view stage's outputs as layers, for the panels and the show: one of its textures as it is."""

# Local application imports
from modules.gl import Texture
from ..layer_base import LayerBase
from .view_layer import ViewLayer


class ViewOutputLayer(LayerBase):
    """One of the view's textures (`ViewOutput`'s, held once by the view), drawn as it is; `timestamp` the view's."""

    def __init__(self, view: ViewLayer, texture: Texture) -> None:
        self._view: ViewLayer = view
        self._texture: Texture = texture

    @property
    def texture(self) -> Texture:
        return self._texture

    @property
    def timestamp(self) -> float | None:
        """The capture time of the capture the view holds, None before the first."""
        return self._view.timestamp

    def deallocate(self) -> None:
        """The view owns the texture."""

    def update(self) -> None:
        """The view makes the texture."""
