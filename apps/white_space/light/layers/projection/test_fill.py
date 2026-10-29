"""TestFill composition — uniform flat colour per channel."""

import numpy as np

from modules.settings import Group

from .._base_layer import ProjectionLayer, ChannelSettings, LayerSettings
from ...frame import Frame


class TestFillSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class TestFill(ProjectionLayer):
    """Fills the entire projection with a flat brightness value on each channel."""

    def __init__(self, resolution: int, settings: TestFillSettings, board) -> None:
        super().__init__(resolution, settings, board)
        self._settings: TestFillSettings = settings

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        white += self._settings.white.level
        blue  += self._settings.blue.level
