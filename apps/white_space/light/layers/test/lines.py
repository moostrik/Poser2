"""Lines composition — discrete scrolling line pattern."""

import math

import numpy as np

from modules.settings import Group

from .._base_layer import BaseLayer, ChannelSettings, LayerSettings
from ...frame import Frame


class LinesSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class Lines(BaseLayer):
    """Discrete bright lines scrolling around the strip."""

    def __init__(self, resolution: int, config: LinesSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._indices: np.ndarray = np.arange(resolution, dtype=np.float32)

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        beat_time = frame.tick.beat + frame.tick.beat_phase
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
