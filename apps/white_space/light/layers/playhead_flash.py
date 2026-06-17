"""PlayheadFlash composition — fires a brief flash each time the rotating
playhead sweeps past an active player's angular position.

When triggered: the first half of the white pixels and all of the blue pixels
light up instantly and fade out over a configurable duration.

Hit detection is centralised in the Compositor and delivered via transport.hits.
"""

from time import monotonic

import numpy as np

from modules.settings import BaseSettings, Field

from ..base import Composition
from ..transport import Transport


class PlayheadFlashSettings(BaseSettings):
    base_white:   Field[float] = Field(0.1,  min=0.0,  max=1.0,  step=0.01, description="Base brightness for white channel (first half of strip)")
    base_blue:    Field[float] = Field(0.1,  min=0.0,  max=1.0,  step=0.01, description="Base brightness for blue channel (full strip)")
    flash_white:  Field[float] = Field(1.0,  min=0.0,  max=1.0,  step=0.01, description="Flash intensity for white channel", newline=True)
    flash_blue:   Field[float] = Field(1.0,  min=0.0,  max=1.0,  step=0.01, description="Flash intensity for blue channel")
    fadeout_time: Field[float] = Field(0.5,  min=0.01, max=10.0, step=0.01, description="Fade-out duration (seconds)", newline=True)


class PlayheadFlash(Composition):
    """Lights the first half of the white strip and all of blue at a base level
    continuously, and adds a fading flash whenever the playhead crosses a player.
    """

    def __init__(self, resolution: int, config: PlayheadFlashSettings) -> None:
        super().__init__(resolution, config)
        self._config = config
        self._flash_start: float = float('-inf')

    # ------------------------------------------------------------------
    # Composition interface
    # ------------------------------------------------------------------

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        now = monotonic()

        if transport.hits:
            self._flash_start = now

        half = self.resolution // 2
        white[:half] += self._config.base_white
        blue[:]      += self._config.base_blue

        elapsed = now - self._flash_start
        fadeout_time = self._config.fadeout_time
        if elapsed >= fadeout_time:
            return

        level_w = float(np.clip(1.0 - elapsed / fadeout_time, 0.0, 1.0))
        white[:half] += level_w * self._config.flash_white
        blue[:]      += level_w * self._config.flash_blue

    def reset(self) -> None:
        self._flash_start = float('-inf')
