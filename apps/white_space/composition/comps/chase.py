"""Chase composition — sine wave scrolling around the strip."""

import math

import numpy as np

from modules.settings import BaseSettings, Group

from ..base import Composition, ChannelSettings
from ..transport import Transport


class ChaseSettings(BaseSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class Chase(Composition):
    """Sine wave chase pattern scrolling continuously around the strip."""

    def __init__(self, resolution: int, config: ChaseSettings) -> None:
        super().__init__(resolution, config)
        self._config = config
        self._indices: np.ndarray = np.arange(resolution, dtype=np.float32)

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        beat_time = transport.beat + transport.phase
        res       = self.resolution
        W         = self._config.white
        B         = self._config.blue

        adj_w    = W.speed * W.amount / 10.0
        phases_w = self._indices * (W.amount * math.tau / res) - beat_time * adj_w * math.tau + W.phase * math.tau
        white   += ((0.5 * np.sin(phases_w) + 0.5) * W.level).astype(white.dtype)

        adj_b    = B.speed * B.amount / 10.0
        phases_b = self._indices * (B.amount * math.tau / res) - beat_time * adj_b * math.tau + B.phase * math.tau
        blue    += ((0.5 * np.sin(phases_b) + 0.5) * B.level).astype(blue.dtype)
