"""HauntedFlash composition — the low-speed player flash. On the WHITE (front-lamp) channel: a
flash as the rotating playhead crosses each *free* live player, and a flash as it crosses each
*ghost* (at the ghost's fixed azimuth, at its own ``ghost_brightness``). A *ghosted* live player
(standing on a spot a ghost claims, tagged ``GhostedFeature = 1`` by ``Ghoster``) gets no white
flash — it's taken over by its ghost — only a BLUE flash a fixed quarter-turn *later*.

The blue lamp trails the white by 0.25 of a turn, so the blue fires when the playhead is a
quarter-turn *past* the pose (``PlayheadOffset ≈ −0.25·2π``, the departing side). Following
``PlayheadFlash``, the window ``width`` and the flash ``brightness`` both interpolate per pose by
its ``PlayheadStability`` from the ``min_*`` endpoints (stability 0) to the ``max_*`` endpoints
(stability 1); the whole-strip level follows the strongest pose. Reuses ``PlayheadFlash``'s
``offset_to_level`` / ``stability_lerp`` kernels. No base level, no gap.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from .playhead_flash import offset_to_level, stability_lerp
from ...frame import Frame
from ....pose import GhostedFeature, PlayheadOffset, PlayheadStability

# The blue lamp sits a quarter-turn behind the white reference, so the blue flash fires when the
# playhead is 0.25 of a turn *past* the pose (PlayheadOffset ≈ −0.25·2π, i.e. departing).
_PHASE_OFFSET: float = 0.25


class HauntedFlashSettings(LayerSettings):
    min_brightness:   Field[float] = Field(0.1, min=0.0, max=1.0,    step=0.01, description="Player flash brightness at stability 0", newline=True)
    max_brightness:   Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="Player flash brightness at stability 1")
    ghost_brightness: Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="Ghost white flash brightness")
    min_width:        Field[float] = Field(20.0, min=0.0, max=360.0, step=1.0, description="Player flash window width (deg) at stability 0", newline=True)
    max_width:        Field[float] = Field(40.0, min=0.0, max=360.0, step=1.0, description="Player flash window width (deg) at stability 1")
    ghost_width:      Field[float] = Field(30.0, min=0.0, max=360.0, step=1.0, description="Ghost flash window width (deg)")


class HauntedFlash(BaseLayer):
    """White flash for each free live player and each ghost at their crossings, plus a blue flash a
    fixed quarter-turn later for ghosted players; width/brightness interpolate by
    ``PlayheadStability`` (ghosts use a fixed ``ghost_brightness``)."""

    def __init__(self, resolution: int, config: HauntedFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        centre: float = _PHASE_OFFSET * math.tau   # blue fires 0.25 turn past the pose (departing side)

        tracklets = self._board.get_tracklets()
        flash_white: float = 0.0
        flash_blue:  float = 0.0
        for pose in self._board.get_frames(self._pose_stage).values():
            tracklet = tracklets.get(pose.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            stability = pose[PlayheadStability].value
            half_rad   = math.radians(stability_lerp(stability, P.min_width, P.max_width) / 2.0)
            brightness = stability_lerp(stability, P.min_brightness, P.max_brightness)
            offset: float = pose[PlayheadOffset].value

            ghosted: float = pose[GhostedFeature].value
            if math.isnan(ghosted) or ghosted < 0.5:
                # WHITE — free (non-ghosted) live player, at the crossing (front lamps).
                w_level = offset_to_level(offset, half_rad)
                if w_level > 0.0:
                    flash_white = max(flash_white, w_level * brightness)
            else:
                # BLUE — ghosted player, a quarter-turn later (wrap the shifted offset to [-π, π)).
                d = offset + centre
                b_level = offset_to_level(math.atan2(math.sin(d), math.cos(d)), half_rad)
                if b_level > 0.0:
                    flash_blue = max(flash_blue, b_level * brightness)

        # WHITE — each ghost flashes at its own (fixed) crossing, at its own brightness/width. Its
        # PlayheadOffset is refreshed to the live playhead by Ghoster, so this tracks the sweep.
        g_half = math.radians(P.ghost_width / 2.0)
        for ghost in self._board.get_ghosts().values():
            g_level = offset_to_level(ghost[PlayheadOffset].value, g_half)
            if g_level > 0.0:
                flash_white = max(flash_white, g_level * P.ghost_brightness)

        half = self.resolution // 2
        if flash_white > 0.0:
            white[:half] += flash_white
        if flash_blue > 0.0:
            blue[:] += flash_blue
