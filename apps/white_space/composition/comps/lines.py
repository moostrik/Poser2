"""Lines composition — discrete scrolling line pattern."""

import math

import numpy as np

from modules.settings import BaseSettings, Group

from ..base import Composition, ChannelSettings
from ..transport import Transport


class LinesSettings(BaseSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class Lines(Composition):
    """Discrete bright lines scrolling around the strip."""

    def __init__(self, resolution: int, config: LinesSettings) -> None:
        super().__init__(resolution, config)
        self._config = config
        self._indices: np.ndarray = np.arange(resolution, dtype=np.float32)

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        beat_time = transport.beat + transport.phase
        res       = self.resolution
        W         = self._config.white
        B         = self._config.blue

        adj_w    = W.speed * W.amount / 10.0
        phases_w = self._indices * (W.amount * math.tau / res) - beat_time * adj_w * math.tau + W.phase * math.tau + math.pi
        vals_w   = 0.5 * np.sin(phases_w) + 0.5
        white   += np.where(vals_w < W.width, W.level, 0.0).astype(white.dtype)

        adj_b    = B.speed * B.amount / 10.0
        phases_b = self._indices * (B.amount * math.tau / res) - beat_time * adj_b * math.tau + B.phase * math.tau + math.pi
        vals_b   = 0.5 * np.sin(phases_b) + 0.5
        blue    += np.where(vals_b < B.width, B.level, 0.0).astype(blue.dtype)
