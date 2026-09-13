"""TestPulse composition — whole-strip brightness oscillating at a uniform rate."""

import math

import numpy as np

from modules.settings import Group

from .._base_layer import ProjectionLayer, ChannelSettings, LayerSettings
from ...frame import Frame


class TestPulseSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class TestPulse(ProjectionLayer):
    """Pulses the whole strip at a uniform sine rate per channel."""

    def __init__(self, resolution: int, config: TestPulseSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        t = frame.tick.time                 # plain seconds; `speed` is the rate in Hz
        W = self._config.white
        B = self._config.blue
        white += (0.5 * math.sin(t * math.tau * W.speed + W.phase * math.tau) + 0.5) * W.level
        blue  += (0.5 * math.sin(t * math.tau * B.speed + B.phase * math.tau) + 0.5) * B.level
