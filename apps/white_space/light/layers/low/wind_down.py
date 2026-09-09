"""WindDown — the dying wall of light: owns the S8/S9 ending fade in both regimes.

The one layer deliberately aware of BOTH light mechanics: it draws the full-strip white
wall, which reads as the POV ring while the machine still spins fast and as the physical
white lamps once it has slowed (see ``low/__init__.py`` for the lamp mapping — when the
low-layer mechanics change, this layer is the single place to update). The states put
the landing look (the playhead line, the sound visuals) underneath at constant weight;
this layer fades the wall to nothing and thereby reveals it.

Fade level: ``f = (1 − ease(elapsed / spin_down_seconds)) × (1 − ease(bars since motor
lock))``. The timed factor is hand-tuned to ride the physical spin-down — its visible
slider is ``statemachine.spin_down_seconds`` (next to ``spin_up_seconds``, its mirror),
shared into this layer's hidden field via the root; the lock factor guarantees the
remainder is extinguished within exactly one round of the reborn playhead after the
motor re-locks at LOW — whichever factor is still unfinished, the wall is gone one bar
after the lock, smoothly and monotonically. ``reset()`` (a show state's entry) restarts
at the full wall. ``progress`` is the read-only fade readout the states' stage_progress
(and S9's sound-visual reveal) ride.
"""

import numpy as np
import pytweening

from modules.settings import Field, Widget

from .._base_layer import LowLayer, LayerSettings
from ...frame import Frame


class WindDownSettings(LayerSettings):
    level:             Field[float] = Field(1.0,  min=0.0, max=1.0,  step=0.01, description="Wall white level (the fade starts here and lands at 0)")
    spin_down_seconds: Field[float] = Field(10.0, min=1.0, max=60.0, step=0.5, visible=False, description="Timed wall fade (seconds) — shared from statemachine.spin_down_seconds, edit it there")
    progress:          Field[float] = Field(0.0,  min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Fade progress (0 = full wall, 1 = gone)")


class WindDown(LowLayer):
    """Draws the wall at the current fade level; see the module docstring."""

    def __init__(self, resolution: int, config: WindDownSettings, board) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._elapsed: float = 0.0
        self._lock_bars: float | None = None   # bar count at the motor lock (None until seen)

    def reset(self) -> None:
        self._elapsed = 0.0
        self._lock_bars = None
        self._config.progress = 0.0

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        self._elapsed += frame.tick.dt
        signals = self._board.get_playhead_signals()
        if self._lock_bars is None and signals.synced:
            self._lock_bars = signals.bars

        t = min(self._elapsed / max(self._config.spin_down_seconds, 1e-6), 1.0)
        f = 1.0 - pytweening.easeInOutSine(t)
        if self._lock_bars is not None:
            b = min(signals.bars - self._lock_bars, 1.0)
            f *= 1.0 - pytweening.easeInOutSine(b)

        self._config.progress = 1.0 - f
        white += f * self._config.level
