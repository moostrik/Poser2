"""TestChase composition — a pulse wave scrolling around the projection, drawn with the light synth's
pulse. At width 0.5 and hardness 0 it is a sine."""

import numpy as np

from modules.settings import Group

from .._base_layer import ProjectionLayer, ChannelSettings, LayerSettings
from ...frame import Frame
from ...synth import Oscillator


class TestChaseSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class TestChase(ProjectionLayer):
    """Chase pattern scrolling continuously around the projection: ``amount`` waves per revolution,
    each ``width`` of its interval wide, flanks by ``hardness`` (``Oscillator.pulse``); width 0.5 and
    hardness 0 is a sine."""

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
        cycle_w  = self._indices * (W.amount / res) - t * adj_w + W.phase    # in cycles: whole numbers at wave crests
        white   += Oscillator.pulse(cycle_w, W.width, W.hardness) * W.level

        adj_b    = B.speed * B.amount / 10.0
        cycle_b  = self._indices * (B.amount / res) - t * adj_b + B.phase
        blue    += Oscillator.pulse(cycle_b, B.width, B.hardness) * B.level
