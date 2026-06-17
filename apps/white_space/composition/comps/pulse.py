"""Pulse composition — whole-strip brightness oscillating at a uniform rate."""

import math

import numpy as np

from modules.settings import BaseSettings, Group

from ..base import Composition, ChannelSettings
from ..transport import Transport


class PulseSettings(BaseSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class Pulse(Composition):
    """Pulses the whole strip at a uniform sine rate per channel."""

    def __init__(self, resolution: int, config: PulseSettings) -> None:
        super().__init__(resolution, config)
        self._config = config

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        beat_time = transport.beat + transport.phase
        W = self._config.white
        B = self._config.blue
        white += (0.5 * math.sin(beat_time * math.tau * W.speed + W.phase * math.tau) + 0.5) * W.level
        blue  += (0.5 * math.sin(beat_time * math.tau * B.speed + B.phase * math.tau) + 0.5) * B.level
