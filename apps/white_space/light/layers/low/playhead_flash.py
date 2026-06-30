"""PlayheadFlash composition — flashes the strip on while the rotating playhead is near each
player, driven by the continuous ``PlayheadOffset``.

Each pose's signed offset to the playhead defines an on/off window around the crossing: the
flash switches on while the playhead is within ``width`` radians of the pose, on either side
of the crossing. Both the window width and the brightness interpolate per pose by its
``PlayheadStability``, from the ``min_*`` endpoints (stability 0) to the ``max_*`` endpoints
(stability 1). The whole-strip brightness follows the closest active pose.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from ...frame import Frame
from ....pose import PlayheadOffset, PlayheadStability


def offset_to_level(phi: float, width: float) -> float:
    """On/off window around the crossing: ``1`` while the playhead is within ``width`` radians
    of the pose (either side of the crossing), ``0`` elsewhere (and for NaN offsets).

    ``phi`` is the pose's signed playhead offset: positive approaching, negative departing.
    """
    if math.isnan(phi):
        return 0.0
    return 1.0 if abs(phi) <= width else 0.0


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
    min_width:  Field[float] = Field(0.2, min=0.0, max=math.pi, step=0.01, description="How far either side of the crossing (rad) the flash is on at stability 0", newline=True)
    max_width:  Field[float] = Field(0.4, min=0.0, max=math.pi, step=0.01, description="How far either side of the crossing (rad) the flash is on at stability 1")


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
            stability = pose[PlayheadStability].value
            width = stability_lerp(stability, P.min_width, P.max_width)
            level = offset_to_level(pose[PlayheadOffset].value, width)
            if level <= 0.0:
                continue
            flash_white = max(flash_white, level * stability_lerp(stability, P.min_white, P.max_white))
            flash_blue  = max(flash_blue,  level * stability_lerp(stability, P.min_blue,  P.max_blue))

        half = self.resolution // 2
        white[:half] += P.base_white + flash_white
        blue[:]      += P.base_blue  + flash_blue
