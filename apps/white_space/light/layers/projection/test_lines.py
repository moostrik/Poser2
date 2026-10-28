"""TestLines composition — discrete scrolling line pattern, drawn with the light synth's pulse."""

import numpy as np

from modules.settings import Group

from .._base_layer import ProjectionLayer, ChannelSettings, LayerSettings
from ...frame import Frame
from ...synth import Oscillator


class TestLinesSettings(LayerSettings):
    white: Group[ChannelSettings] = Group(ChannelSettings)
    blue:  Group[ChannelSettings] = Group(ChannelSettings)


class TestLines(ProjectionLayer):
    """Bright lines scrolling around the projection: ``amount`` lines per revolution, each ``width``
    of its interval wide, flanks by ``hardness`` (``Oscillator.pulse``)."""

    def __init__(self, resolution: int, config: TestLinesSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._indices: np.ndarray = np.arange(resolution, dtype=np.float32)

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        t         = frame.tick.time         # plain seconds; `speed` scales the rate
        res       = self.resolution
        W         = self._config.white
        B         = self._config.blue

        adj_w    = W.speed * W.amount / 10.0
        cycle_w  = self._indices * (W.amount / res) - t * adj_w + W.phase    # in cycles: whole numbers at line centres
        white   += Oscillator.pulse(cycle_w, W.width, W.hardness) * W.level

        adj_b    = B.speed * B.amount / 10.0
        cycle_b  = self._indices * (B.amount / res) - t * adj_b + B.phase
        blue    += Oscillator.pulse(cycle_b, B.width, B.hardness) * B.level
