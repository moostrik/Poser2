from enum import IntEnum
from typing import Callable, Generic, TypeVar

from modules.gl import Fbo, Text, clear_color
from modules.pose import BaseScalarFeature
from modules.utils import Color, Rect

from ...color_settings import ColorSettings
from ...layer_base import DataCache
from .data_layer_settings import DataLayerSettingsProtocol, LayerMode, TRACK_COLOR_FEATURES

T = TypeVar('T')

# Where label i of n sits in a plot of the given rect: its text's (x, y) in pixels
LabelPlace = Callable[[int, int, Rect], tuple[int, int]]


class FeaturePlot(Generic[T]):
    """What a feature layer draws into and with: its mode, its colors, its plot and label Fbos, the data it last drew
    and its labels."""

    def __init__(self, mode: LayerMode, settings: DataLayerSettingsProtocol, color_settings: ColorSettings,
                 feature_map: dict[IntEnum, type[BaseScalarFeature]], draw_labels: bool, font_size: int) -> None:
        self._mode: LayerMode = mode
        self.settings: DataLayerSettingsProtocol = settings
        self._color_settings: ColorSettings = color_settings
        self.feature_map: dict[IntEnum, type[BaseScalarFeature]] = feature_map
        self._font_size: int = font_size
        self._was_active: bool = False

        self.fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self.data_cache: DataCache[T] = DataCache[T]()
        self._labels: list[str] = []
        self.draw_labels: bool = draw_labels
        self._text_renderer: Text = Text()

    @property
    def active(self) -> bool:
        """Whether the settings' mode is this plot's."""
        return self.settings.mode == self._mode

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)
        self._text_renderer.allocate(font_size=self._font_size)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self._label_fbo.deallocate()
        self._text_renderer.deallocate()

    def clear(self) -> None:
        """Forget the data it last drew and its labels."""
        self.data_cache = DataCache[T]()
        self._labels = []

    def update_active(self) -> bool:
        """Whether it is active this tick; it clears when it stops being active."""
        active = self.active
        if self._was_active and not active:
            self.clear()
        self._was_active = active
        return active

    def draw(self) -> None:
        if not self.active:
            return
        if self.fbo.allocated:
            self.fbo.draw()
            if self.draw_labels and self.settings.render_labels:
                self._label_fbo.draw()

    def colors(self) -> list[tuple[float, float, float, float]]:
        """The line colors: the history color, the track colors for a feature drawn per track, or the default colors."""
        if self.settings.use_history_color:
            return [self._color_settings.history.to_tuple()]
        feature_type = self.feature_map[self.settings.feature_field]
        if feature_type in TRACK_COLOR_FEATURES:
            return self._color_settings.track_color_tuples
        return self._color_settings.default_color_tuples

    def update_labels(self, labels: list[str], place: LabelPlace) -> None:
        """Draw the labels, each at its place, when the settings ask for them and they changed."""
        if not self.settings.render_labels or labels == self._labels:
            return
        fbo = self._label_fbo
        rect = Rect(0, 0, fbo.width, fbo.height)
        colors = self.colors()

        fbo.begin()
        clear_color()
        for i, string in enumerate(labels):
            x, y = place(i, len(labels), rect)
            self._text_renderer.draw_box_text(
                x, y, string, Color(*colors[i % len(colors)]), Color(0.0, 0.0, 0.0, 0.66),
                screen_width=fbo.width, screen_height=fbo.height
            )
        fbo.end()
        self._labels = labels
