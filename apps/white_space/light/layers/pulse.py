"""Pulse composition — whole-strip brightness oscillating at a uniform rate."""

import math

import numpy as np

from modules.settings import Group

from ._base_layer import BaseLayer, ChannelSettings, LayerSettings
from ..frame import Frame


class PulseSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class Pulse(BaseLayer):
    """Pulses the whole strip at a uniform sine rate per channel."""

    def __init__(self, resolution: int, config: PulseSettings) -> None:
        super().__init__(resolution, config)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        beat_time = frame.tick.beat + frame.tick.phase
        W = self._config.white
        B = self._config.blue
        white += (0.5 * math.sin(beat_time * math.tau * W.speed + W.phase * math.tau) + 0.5) * W.level
        blue  += (0.5 * math.sin(beat_time * math.tau * B.speed + B.phase * math.tau) + 0.5) * B.level
