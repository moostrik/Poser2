"""PlayheadFlash composition — flashes the strip on while the rotating playhead is near each
player, driven by the continuous ``PlayheadOffset``.

Each pose's signed offset to the playhead defines an on/off window around the crossing: the
flash switches on while the playhead is within ``rise`` radians before the pose and stays on
until ``fall`` radians after it — an asymmetric lead/trail a single hit moment could never
express. The whole-strip brightness follows the closest active pose.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from ...frame import Frame
from ....pose import PlayheadOffset, PlayheadStability


def offset_to_level(phi: float, rise: float, fall: float) -> float:
    """On/off window around the crossing: ``1`` while the playhead is within ``rise`` radians
    before the pose or ``fall`` radians after it, ``0`` elsewhere (and for NaN offsets).

    ``phi`` is the pose's signed playhead offset: positive approaching, negative departing.
    """
    if math.isnan(phi):
        return 0.0
    return 1.0 if -fall <= phi <= rise else 0.0


def stability_to_flash(stability: float, under: float, flash: float) -> float:
    """Linear map of a pose's playhead stability to peak flash intensity.

    Stability ``0.0`` (or NaN — no playhead hit yet / after an azimuth reset) maps to the
    settable ``under`` floor; stability ``1.0`` maps to the full ``flash`` value.
    """
    if math.isnan(stability):
        stability = 0.0
    return under + stability * (flash - under)


class PlayheadFlashSettings(LayerSettings):
    base_white:  Field[float] = Field(0.1, min=0.0, max=1.0, step=0.01, description="Base brightness for white channel (first half of strip)")
    base_blue:   Field[float] = Field(0.1, min=0.0, max=1.0, step=0.01, description="Base brightness for blue channel (full strip)")
    flash_white: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Peak flash intensity for white channel at stability 1", newline=True)
    flash_blue:  Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Peak flash intensity for blue channel at stability 1")
    under_white: Field[float] = Field(0.2, min=0.0, max=1.0, step=0.01, description="Flash intensity floor for white channel at stability 0", newline=True)
    under_blue:  Field[float] = Field(0.2, min=0.0, max=1.0, step=0.01, description="Flash intensity floor for blue channel at stability 0")
    rise:        Field[float] = Field(0.5, min=0.0, max=math.pi, step=0.01, description="How far ahead of the crossing (radians) the flash switches on", newline=True)
    fall:        Field[float] = Field(0.5, min=0.0, max=math.pi, step=0.01, description="How far behind the crossing (radians) the flash switches off")


class PlayheadFlash(BaseLayer):
    """Continuous base level plus an on/off flash window tracking the playhead's approach to
    each active player, read from ``PlayheadOffset``. Each pose's peak flash intensity is
    scaled by its ``PlayheadStability``: a steady pose flashes up to the full ``flash_*``
    value, an unstable one only up to the ``under_*`` floor."""

    def __init__(self, resolution: int, config: PlayheadFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        rise: float = P.rise
        fall: float = P.fall

        tracklets = self._board.get_tracklets()
        flash_white: float = 0.0
        flash_blue:  float = 0.0
        for pose in self._board.get_frames(self._pose_stage).values():
            tracklet = tracklets.get(pose.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            level = offset_to_level(pose[PlayheadOffset].value, rise, fall)
            if level <= 0.0:
                continue
            stability = pose[PlayheadStability].value
            flash_white = max(flash_white, level * stability_to_flash(stability, P.under_white, P.flash_white))
            flash_blue  = max(flash_blue,  level * stability_to_flash(stability, P.under_blue,  P.flash_blue))

        half = self.resolution // 2
        white[:half] += P.base_white + flash_white
        blue[:]      += P.base_blue  + flash_blue
