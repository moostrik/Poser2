"""PlayheadFlash composition — flashes the strip on while the rotating playhead is near each
player, driven by the continuous ``PlayheadOffset``.

Each pose's signed offset to the playhead defines an on/off window around the crossing: the
flash switches on while the playhead is within the ``width``° window of the crossing
(±``width``/2 each side), save a dark ``gap`` notch straddling the crossing itself. Both the window
width and the brightness interpolate per pose by its ``PlayheadStability``, from the ``min_*``
endpoints (stability 0) to the ``max_*`` endpoints (stability 1). The whole-strip brightness
follows the closest active pose.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from ...frame import Frame
from ....pose import PlayheadElement, PlayheadOffset, PlayheadStability


def offset_to_level(phi: float, width: float, gap: float = 0.0) -> float:
    """On/off window around the crossing: ``1`` while the playhead is within ``width`` radians
    of the pose, except the central ``gap`` fraction of that width — a dark notch straddling
    the crossing. ``0`` outside the window, inside the notch, and for NaN offsets.

    ``phi`` is the pose's signed playhead offset: positive approaching, negative departing.
    """
    if math.isnan(phi):
        return 0.0
    distance = abs(phi)
    return 1.0 if gap * width <= distance <= width else 0.0


def stability_lerp(stability: float, lo: float, hi: float) -> float:
    """Linear map of a pose's playhead stability to a value endpoint pair.

    Stability ``0.0`` (or NaN — no playhead hit yet / after an azimuth reset) maps to ``lo``;
    stability ``1.0`` maps to ``hi``. Used for both flash intensities and window widths.
    """
    if math.isnan(stability):
        stability = 0.0
    return lo + stability * (hi - lo)


class PlayheadFlashSettings(LayerSettings):
    base_white: Field[float] = Field(0.0, min=0.0, max=1.0,     step=0.01, description="Base brightness for white channel (first half of strip)")
    base_blue:  Field[float] = Field(0.0, min=0.0, max=1.0,     step=0.01, description="Base brightness for blue channel (full strip)")
    min_white:  Field[float] = Field(0.1, min=0.0, max=1.0,     step=0.01, description="White flash intensity at stability 0", newline=True)
    max_white:  Field[float] = Field(1.0, min=0.0, max=1.0,     step=0.01, description="White flash intensity at stability 1")
    min_blue:   Field[float] = Field(0.1, min=0.0, max=1.0,     step=0.01, description="Blue flash intensity at stability 0", newline=True)
    max_blue:   Field[float] = Field(1.0, min=0.0, max=1.0,     step=0.01, description="Blue flash intensity at stability 1")
    min_width:  Field[float] = Field(20.0, min=0.0, max=360.0, step=1.0, description="Flash window width (deg) at stability 0", newline=True)
    max_width:  Field[float] = Field(40.0, min=0.0, max=360.0, step=1.0, description="Flash window width (deg) at stability 1")
    gap:        Field[float] = Field(0.25, min=0.0, max=1.0,    step=0.01, description="Fraction of the window centre that stays dark — a notch at the crossing (0 = solid)", newline=True)


class PlayheadFlash(BaseLayer):
    """Continuous base level plus an on/off flash window tracking the playhead's approach to
    each active player, read from ``PlayheadOffset``. Each pose's window width and flash
    intensity are interpolated by its ``PlayheadStability`` from the ``min_*`` endpoints
    (stability 0) to the ``max_*`` endpoints (stability 1): a steady pose flashes brighter
    and over a wider window than an unstable one."""

    def __init__(self, resolution: int, config: PlayheadFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config

        tracklets = self._board.get_tracklets()
        flash_white: float = 0.0
        flash_blue:  float = 0.0
        for pose in self._board.get_frames(self._pose_stage).values():
            tracklet = tracklets.get(pose.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            stability = pose[PlayheadStability].get(PlayheadElement.Stability)
            half_rad = math.radians(stability_lerp(stability, P.min_width, P.max_width) / 2.0)
            level = offset_to_level(pose[PlayheadOffset].value, half_rad, P.gap)
            if level <= 0.0:
                continue
            flash_white = max(flash_white, level * stability_lerp(stability, P.min_white, P.max_white))
            flash_blue  = max(flash_blue,  level * stability_lerp(stability, P.min_blue,  P.max_blue))

        half = self.resolution // 2
        white[:half] += P.base_white + flash_white
        blue[:]      += P.base_blue  + flash_blue
