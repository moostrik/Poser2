"""Fill composition — uniform flat colour per channel."""

import numpy as np

from modules.settings import Group

from ..base_layer import BaseLayer, ChannelSettings, LayerSettings
from ..frame import Frame


class FillSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class Fill(BaseLayer):
    """Fills the entire strip with a flat brightness value on each channel."""

    def __init__(self, resolution: int, config: FillSettings) -> None:
        super().__init__(resolution, config)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        white += self._config.white.level
        blue  += self._config.blue.level
