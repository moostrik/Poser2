"""TestChase composition — sine wave scrolling around the strip."""

import math

import numpy as np

from modules.settings import Group

from .._base_layer import ProjectionLayer, ChannelSettings, LayerSettings
from ...frame import Frame


class TestChaseSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class TestChase(ProjectionLayer):
    """Sine wave chase pattern scrolling continuously around the strip."""

    def __init__(self, resolution: int, config: TestChaseSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._indices: np.ndarray = np.arange(resolution, dtype=np.float32)

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        t         = frame.tick.time         # plain seconds; `speed` scales the rate
        res       = self.resolution
        W         = self._config.white
        B         = self._config.blue

        adj_w    = W.speed * W.amount / 10.0
        phases_w = self._indices * (W.amount * math.tau / res) - t * adj_w * math.tau + W.phase * math.tau
        white   += ((0.5 * np.sin(phases_w) + 0.5) * W.level).astype(white.dtype)

        adj_b    = B.speed * B.amount / 10.0
        phases_b = self._indices * (B.amount * math.tau / res) - t * adj_b * math.tau + B.phase * math.tau
        blue    += ((0.5 * np.sin(phases_b) + 0.5) * B.level).astype(blue.dtype)
