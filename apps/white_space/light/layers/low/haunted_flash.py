"""HauntedFlash composition — the low-speed player flash, all on the WHITE (front-lamp) channel.
A flash as the rotating playhead crosses each live player, and a flash as it crosses each *ghost*
(at the ghost's fixed azimuth, its own ``ghost_brightness`` / ``ghost_width``). Live persons and
ghosts are independent flashes — no muting, no blue.

Player flashes interpolate their window ``width`` by ``PlayheadStability`` and their ``brightness`` by
``PlayheadMotion``, each from the ``min_*`` (value 0) to ``max_*`` (value 1) endpoints; ghosts use a
fixed ``ghost_brightness`` / ``ghost_width``. Reuses ``PlayheadFlash``'s ``offset_to_level`` /
``stability_lerp`` kernels. No base level, no gap.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from .playhead_flash import offset_to_level, stability_lerp
from ...frame import Frame
from ....pose import Fade, PlayheadElement, PlayheadOffset, PlayheadStability


class HauntedFlashSettings(LayerSettings):
    min_brightness:   Field[float] = Field(0.1, min=0.0, max=1.0,    step=0.01, description="Player flash brightness at motion 0", newline=True)
    max_brightness:   Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="Player flash brightness at motion 1")
    ghost_brightness: Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="Ghost flash brightness")
    min_width:        Field[float] = Field(20.0, min=0.0, max=360.0, step=1.0, description="Player flash window width (deg) at stability 0", newline=True)
    max_width:        Field[float] = Field(40.0, min=0.0, max=360.0, step=1.0, description="Player flash window width (deg) at stability 1")
    ghost_width:      Field[float] = Field(30.0, min=0.0, max=360.0, step=1.0, description="Ghost flash window width (deg)")


class HauntedFlash(BaseLayer):
    """White flash as the playhead crosses each live player (``PlayheadStability`` width /
    ``PlayheadMotion`` brightness) and each ghost (fixed ``ghost_brightness`` / ``ghost_width``)."""

    def __init__(self, resolution: int, config: HauntedFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config

        tracklets = self._board.get_tracklets()
        flash: float = 0.0

        # Live players — window by PlayheadStability, brightness by PlayheadMotion.
        for pose in self._board.get_frames(self._pose_stage).values():
            tracklet = tracklets.get(pose.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            stability  = pose[PlayheadStability].get(PlayheadElement.Stability)
            motion     = pose[PlayheadStability].get(PlayheadElement.Motion)
            half_rad   = math.radians(stability_lerp(stability, P.min_width, P.max_width) / 2.0)
            brightness = stability_lerp(motion, P.min_brightness, P.max_brightness)
            level = offset_to_level(pose[PlayheadOffset].value, half_rad)
            if level > 0.0:
                flash = max(flash, level * brightness)

        # Ghosts — flash at their own (fixed) crossing; PlayheadOffset is refreshed by Ghoster.
        g_half = math.radians(P.ghost_width / 2.0)
        for ghost in self._board.get_ghosts().values():
            level = offset_to_level(ghost[PlayheadOffset].value, g_half)
            if level > 0.0:
                fade = ghost[Fade].value if Fade in ghost else 1.0
                flash = max(flash, level * P.ghost_brightness * fade)

        if flash > 0.0:
            white[:self.resolution // 2] += flash
