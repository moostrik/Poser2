"""HauntedFlash composition — the low-speed player/ghost flash.

WHITE (front-lamp) channel: a flash as the rotating playhead crosses each live player (window width by
``PlayheadStability`` / brightness by its motion), and a flash as it crosses each **active** ghost (at the
ghost's fixed azimuth, its own ``ghost_brightness`` / ``ghost_width``, dimmed by its ``Fade``).

BLUE (full-strip) channel: a flash for each **passive** ghost — a ghost still forming on its live person —
fired a fixed quarter-turn *later* than the crossing (``PlayheadOffset ≈ −0.25·2π``, the departing side),
so it trails the white. Its window widens with the passive ghost's ``Dwell`` and brightens with its
``Motion``, so a ghost about to break free glows harder. Reuses ``PlayheadFlash``'s ``offset_to_level`` /
``stability_lerp`` kernels. No base level, no gap.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from .playhead_flash import offset_to_level, stability_lerp
from ...frame import Frame
from ....pose import Fade, GhostStateValue, PlayheadElement, PlayheadOffset, PlayheadStability, ghost_state

# The blue lamp trails the white by a quarter-turn, so the blue flash fires when the playhead is 0.25 of
# a turn *past* the passive ghost (PlayheadOffset ≈ −0.25·2π, i.e. departing).
_PHASE_OFFSET: float = 0.25


class HauntedFlashSettings(LayerSettings):
    min_brightness:   Field[float] = Field(0.1, min=0.0, max=1.0,    step=0.01, description="Player/passive flash brightness at motion 0", newline=True)
    max_brightness:   Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="Player/passive flash brightness at motion 1")
    ghost_brightness: Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="Active ghost flash brightness")
    min_width:        Field[float] = Field(20.0, min=0.0, max=360.0, step=1.0, description="Player/passive flash window width (deg) at stability/dwell 0", newline=True)
    max_width:        Field[float] = Field(40.0, min=0.0, max=360.0, step=1.0, description="Player/passive flash window width (deg) at stability/dwell 1")
    ghost_width:      Field[float] = Field(30.0, min=0.0, max=360.0, step=1.0, description="Active ghost flash window width (deg)")


class HauntedFlash(BaseLayer):
    """White flash as the playhead crosses each live player and each **active** ghost, plus a blue flash a
    quarter-turn later for each **passive** ghost (width by its Dwell, brightness by its Motion)."""

    def __init__(self, resolution: int, config: HauntedFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        centre: float = _PHASE_OFFSET * math.tau   # blue fires 0.25 turn past the passive ghost

        tracklets = self._board.get_tracklets()
        flash_white: float = 0.0
        flash_blue:  float = 0.0

        # WHITE — live players: window by PlayheadStability, brightness by PlayheadMotion.
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
                flash_white = max(flash_white, level * brightness)

        # Ghosts — ACTIVE flash white at their fixed crossing; PASSIVE flash blue a quarter-turn later.
        g_half = math.radians(P.ghost_width / 2.0)
        for ghost in self._board.get_ghosts().values():
            state = ghost_state(ghost)
            offset = ghost[PlayheadOffset].value
            if state is GhostStateValue.ACTIVE:
                level = offset_to_level(offset, g_half)
                if level > 0.0:
                    fade = ghost[Fade].value if Fade in ghost else 1.0
                    flash_white = max(flash_white, level * P.ghost_brightness * fade)
            elif state is GhostStateValue.PASSIVE:
                stab = ghost[PlayheadStability]
                half_rad   = math.radians(stability_lerp(stab.get(PlayheadElement.Dwell), P.min_width, P.max_width) / 2.0)
                brightness = stability_lerp(stab.get(PlayheadElement.Motion), P.min_brightness, P.max_brightness)
                d = offset + centre   # shift a quarter-turn, then wrap to [-π, π)
                level = offset_to_level(math.atan2(math.sin(d), math.cos(d)), half_rad)
                if level > 0.0:
                    flash_blue = max(flash_blue, level * brightness)

        half = self.resolution // 2
        if flash_white > 0.0:
            white[:half] += flash_white
        if flash_blue > 0.0:
            blue[:] += flash_blue
