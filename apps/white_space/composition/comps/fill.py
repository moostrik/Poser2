"""Fill composition — uniform flat colour per channel."""

import numpy as np

from modules.settings import BaseSettings, Field, Group

from ..base import Composition, ChannelSettings
from ..transport import Transport


class FillSettings(BaseSettings):
    enabled: Field[bool]  = Field(False, description="Enable Fill composition")
    gain:    Field[float] = Field(1.0,   min=0.0, max=2.0, step=0.01, description="Output gain")
    white:   Group[ChannelSettings] = Group(ChannelSettings)
    blue:    Group[ChannelSettings] = Group(ChannelSettings)


class Fill(Composition):
    """Fills the entire strip with a flat brightness value on each channel."""

    def __init__(self, resolution: int, config: FillSettings) -> None:
        super().__init__(resolution, config)
        self._config = config

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        if not self._config.enabled:
            return
        g = self._config.gain
        white += self._config.white.strength * g
        blue  += self._config.blue.strength  * g
