"""Fill composition — uniform flat colour per channel."""

import numpy as np

from modules.settings import BaseSettings, Group

from ..base import Composition, ChannelSettings
from ..transport import Transport


class FillSettings(BaseSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class Fill(Composition):
    """Fills the entire strip with a flat brightness value on each channel."""

    def __init__(self, resolution: int, config: FillSettings) -> None:
        super().__init__(resolution, config)
        self._config = config

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        white += self._config.white.level
        blue  += self._config.blue.level
