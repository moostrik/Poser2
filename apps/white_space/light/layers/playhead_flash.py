"""PlayheadFlash composition — brightens the strip as the rotating playhead approaches
each player and fades as it departs, driven by the continuous ``PlayheadPhase``.

Replaces the old discrete hit-flash: each pose's signed phase to the playhead gives a
smooth rise (approaching, phase ``+rise → 0``) and fall (departing, ``0 → -fall``) around
the crossing — the asymmetric lead/trail a single hit moment could never express. The
whole-strip brightness follows the closest-approaching pose.
"""

import math

import numpy as np

from modules.settings import Field

from ._base_layer import BaseLayer, LayerSettings
from ..frame import Frame
from ...pose import PlayheadPhase


def phase_to_level(phi: float, rise: float, fall: float) -> float:
    """Asymmetric triangular envelope of a pose's playhead phase, peaking at the crossing.

    ``rise``/``fall`` are widths in radians. Returns 0 for NaN or out-of-window phases.
    """
    if math.isnan(phi):
        return 0.0
    if rise > 0.0 and 0.0 <= phi <= rise:
        return 1.0 - phi / rise          # approaching: 0 at φ=rise → 1 at φ=0
    if fall > 0.0 and -fall <= phi < 0.0:
        return 1.0 + phi / fall          # departing: 1 at φ=0 → 0 at φ=-fall
    return 0.0


class PlayheadFlashSettings(LayerSettings):
    base_white:  Field[float] = Field(0.1, min=0.0, max=1.0, step=0.01, description="Base brightness for white channel (first half of strip)")
    base_blue:   Field[float] = Field(0.1, min=0.0, max=1.0, step=0.01, description="Base brightness for blue channel (full strip)")
    flash_white: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Peak flash intensity for white channel", newline=True)
    flash_blue:  Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Peak flash intensity for blue channel")
    rise:        Field[float] = Field(0.5, min=0.0, max=math.pi, step=0.01, description="Approach width (radians): phase ahead at which brightening starts", newline=True)
    fall:        Field[float] = Field(0.5, min=0.0, max=math.pi, step=0.01, description="Departure width (radians): phase behind at which the glow fades out")


class PlayheadFlash(BaseLayer):
    """Continuous base level plus a rise/fall glow tracking the playhead's approach to
    each active player, read from ``PlayheadPhase``."""

    def __init__(self, resolution: int, config: PlayheadFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        rise: float = P.rise
        fall: float = P.fall

        tracklets = self._board.get_tracklets()
        level: float = 0.0
        for pose in self._board.get_frames(self._pose_stage).values():
            tracklet = tracklets.get(pose.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            level = max(level, phase_to_level(pose[PlayheadPhase].value, rise, fall))

        half = self.resolution // 2
        white[:half] += P.base_white + level * P.flash_white
        blue[:]      += P.base_blue  + level * P.flash_blue
