"""PlayheadFlash composition — fires a brief flash each time the rotating
playhead sweeps past an active player's angular position.

When triggered: the first half of the white pixels and all of the blue pixels
light up instantly and fade out over a configurable duration.

Trigger logic: the playhead sweeps continuously in one direction (0 → 1 → 0 …).
Each tick the interval (prev_playhead, curr_playhead] is checked for containment
of each player's strip_pos, with correct modular wrap-around handling.  Any
active player crossing resets the single shared fade timer.
"""

from time import monotonic

import numpy as np

from modules.settings import BaseSettings, Field
from modules.tracker import Tracklet
from modules.pose.frame import Frame
from modules.pose import features

from ..base import Composition
from ..transport import Transport, MotorMode


class PlayheadFlashSettings(BaseSettings):
    base_white:   Field[float] = Field(0.1,  min=0.0,  max=1.0,  step=0.01, description="Base brightness for white channel (first half of strip)")
    base_blue:    Field[float] = Field(0.1,  min=0.0,  max=1.0,  step=0.01, description="Base brightness for blue channel (full strip)")
    flash_white:  Field[float] = Field(1.0,  min=0.0,  max=1.0,  step=0.01, description="Flash intensity for white channel", newline=True)
    flash_blue:   Field[float] = Field(1.0,  min=0.0,  max=1.0,  step=0.01, description="Flash intensity for blue channel")
    fadeout_time: Field[float] = Field(0.5,  min=0.01, max=10.0, step=0.01, description="Fade-out duration (seconds)", newline=True)


def _sweep_contains(prev: float, curr: float, pos: float) -> bool:
    """Return True if position *pos* falls inside the half-open interval
    (prev, curr] on the unit circle, accounting for wrap-around.

    Assumes the playhead advances in the direction of increasing value (modulo 1).
    """
    if prev <= curr:
        return prev < pos <= curr
    else:
        # Wrap: interval spans the 0 boundary (e.g. 0.95 → 0.05)
        return pos > prev or pos <= curr


class PlayheadFlash(Composition):
    """Lights the first half of the white strip and all of blue at a base level
    continuously, and adds a fading flash whenever the playhead crosses a player.
    """

    def __init__(self, resolution: int, config: PlayheadFlashSettings) -> None:
        super().__init__(resolution, config)
        self._config = config
        self._player_positions: list[float] = []
        self._flash_start:   float = float('-inf')
        self._prev_playhead: float = 0.0
        self._last_render:   float = float('-inf')  # monotonic time of last render call

    # ------------------------------------------------------------------
    # Composition interface
    # ------------------------------------------------------------------

    def set_pose_inputs(self, frames: list[Frame], tracklets: dict[int, Tracklet]) -> None:
        positions: list[float] = []
        for frame in frames:
            tracklet = tracklets.get(frame.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            strip_pos: float = frame[features.Azimuth].value
            if not np.isnan(strip_pos):
                positions.append(strip_pos)
        self._player_positions = positions

    def render(self, transport: Transport, white: np.ndarray, blue: np.ndarray) -> None:
        now  = monotonic()
        curr = transport.playhead

        # Only use the stored prev if we rendered last tick; otherwise the comp
        # was inactive and _prev_playhead is stale — skip the crossing check.
        # Use a fixed 0.5 s threshold: safely above any realistic tick interval.
        gap_ok = (now - self._last_render) < 0.5

        if gap_ok and transport.motor_mode == MotorMode.LOW_SPEED:
            prev = self._prev_playhead
            for pos in self._player_positions:
                if _sweep_contains(prev, curr, pos):
                    self._flash_start = now
                    break

        self._prev_playhead = curr
        self._last_render   = now

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
        self._flash_start   = float('-inf')
        self._prev_playhead = 0.0
        self._last_render   = float('-inf')
