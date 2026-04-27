"""Pulse composition — whole-strip brightness oscillating at a uniform rate."""

import math

import numpy as np

from modules.settings import BaseSettings, Field, Group

from ..base import Composition, ChannelSettings
from ..transport import Transport


class PulseSettings(BaseSettings):
    enabled: Field[bool]  = Field(False, description="Enable Pulse composition")
    gain:    Field[float] = Field(1.0,   min=0.0, max=2.0, step=0.01, description="Output gain")
    white:   Group[ChannelSettings] = Group(ChannelSettings)
    blue:    Group[ChannelSettings] = Group(ChannelSettings)


class Pulse(Composition):
    """Pulses the whole strip at a uniform sine rate per channel."""

    def __init__(self, resolution: int, config: PulseSettings) -> None:
        super().__init__(resolution, config)
        self._config = config

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        if not self._config.enabled:
            return
        g = self._config.gain
        beat_time = transport.beat + transport.phase  # continuously increasing in beats
        W = self._config.white
        B = self._config.blue
        white += (0.5 * math.sin(beat_time * math.tau * W.speed + W.phase * math.tau) + 0.5) * W.strength * g
        blue  += (0.5 * math.sin(beat_time * math.tau * B.speed + B.phase * math.tau) + 0.5) * B.strength * g
