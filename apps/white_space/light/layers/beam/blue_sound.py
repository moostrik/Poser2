"""BlueSound — the soundscape made visible: the left and right blue lamps follow the
actual levels Max is playing (``/WS/sound/level`` → the board's sound-level store).

Latency first: no envelope shaping — Max owns the envelope, the lamps follow. The only
smoothing is a window of at most a few light frames (``smoothing_frames``), there purely
to bridge OSC-arrival vs tick timing jitter. When no message has arrived for
``stale_seconds`` the layer falls back (off, or a gentle idle pulse) so a silent or
disconnected Max never freezes the lamps at a stuck level.
"""

from __future__ import annotations

import math
from collections import deque
from enum import IntEnum, auto
from time import monotonic
from typing import TYPE_CHECKING

import numpy as np

from modules.settings import Field

from .._base_layer import BeamLayer, LayerSettings
from ...frame import Frame

if TYPE_CHECKING:
    from ....board import Board

# The gentle idle pulse's rate while input is stale (slow breath, ~12 cycles/minute).
_PULSE_HZ: float = 0.2


class SoundFallback(IntEnum):
    """What the blue lamps do when the sound input has gone stale."""
    OFF   = 0
    PULSE = auto()


class BlueSoundSettings(LayerSettings):
    gain:             Field[float]         = Field(1.0, min=0.0, max=2.0, step=0.01, description="Level → lamp gain")
    smoothing_frames: Field[int]           = Field(2,   min=0,   max=3,   step=1,    description="Jitter bridge only: average over at most this many light frames (0 = raw)")
    stale_seconds:    Field[float]         = Field(2.0, min=0.1, max=30.0, step=0.1, description="No message for this long → fall back", newline=True)
    fallback:         Field[SoundFallback] = Field(SoundFallback.OFF, description="Stale input: lamps off, or a gentle idle pulse")
    fallback_level:   Field[float]         = Field(0.15, min=0.0, max=1.0, step=0.01, description="Idle pulse peak level")


class BlueSound(BeamLayer):
    """Left level → left blue lamp, right level → right blue lamp; see the module docstring."""

    def __init__(self, resolution: int, config: BlueSoundSettings, board: Board) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._window: deque[tuple[float, float]] = deque(maxlen=3)

    def reset(self) -> None:
        """Clear the smoothing window — the lamps re-attack from the live levels."""
        self._window.clear()

    def _draw(self, frame: Frame, beam_lights: np.ndarray) -> None:
        P = self._config
        levels = self._board.get_sound_levels()

        if levels.timestamp == 0.0 or monotonic() - levels.timestamp > P.stale_seconds:
            self._window.clear()
            if P.fallback == SoundFallback.PULSE:
                pulse = P.fallback_level * (0.5 - 0.5 * math.cos(math.tau * _PULSE_HZ * frame.tick.time))
                self._add_beam_lights(beam_lights, left_blue=pulse, right_blue=pulse)
            return

        self._window.append((levels.left, levels.right))
        n = max(1, min(int(P.smoothing_frames), len(self._window)))
        recent = list(self._window)[-n:]
        left  = sum(v[0] for v in recent) / n
        right = sum(v[1] for v in recent) / n
        self._add_beam_lights(beam_lights, left_blue=left * P.gain, right_blue=right * P.gain)
