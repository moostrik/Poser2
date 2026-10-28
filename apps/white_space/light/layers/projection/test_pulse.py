"""TestPulse composition — the whole projection's brightness pulsing at a uniform rate, drawn with the
light synth's pulse in time. At width 0.5 and hardness 0 it is a sine; at hardness 1 an on/off strobe
lit for ``width`` of every cycle."""

import numpy as np

from modules.settings import Group

from .._base_layer import ProjectionLayer, ChannelSettings, LayerSettings
from ...frame import Frame
from ...synth import Oscillator


class TestPulseSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class TestPulse(ProjectionLayer):
    """Pulses the whole projection at a uniform rate per channel: ``speed`` cycles per second, lit
    for ``width`` of every cycle, flanks by ``hardness`` (``Oscillator.pulse``)."""

    def __init__(self, resolution: int, config: TestPulseSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        t = frame.tick.time                 # plain seconds; `speed` is the rate in Hz
        W = self._config.white
        B = self._config.blue
        white += Oscillator.pulse(np.asarray(t * W.speed + W.phase), W.width, W.hardness) * W.level
        blue  += Oscillator.pulse(np.asarray(t * B.speed + B.phase), B.width, B.hardness) * B.level
