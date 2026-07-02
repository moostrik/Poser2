"""HauntedFlash composition — the low-speed player/ghost flash.

WHITE (front-lamp) channel: a flash as the rotating playhead crosses each live player (at full
``white``), and — when ``ghosts`` is enabled — as it crosses each **active** ghost at its fixed
azimuth, dimmed by that ghost's ``GhostFeature`` Fade (a released ghost decays 1→0).

BLUE (front-lamp) channel: a flash for each **verified** passive ghost — one whose ``Dwell`` and
``Motion`` have both reached ``1.0`` (settled = ready to break free into an active ghost) — fired a
fixed quarter-turn *later* than the crossing (``PlayheadOffset ≈ −0.25·2π``, the departing side),
so it trails the white. A still-building passive ghost gets no blue. Fade is irrelevant to blue.

A single ``width`` sizes every window; ``white`` / ``blue`` set the two brightnesses. Dwell/Motion
no longer shape the flash. Reuses ``PlayheadFlash``'s ``offset_to_level`` kernel. No base, no gap.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BaseLayer, LayerSettings
from .playhead_flash import offset_to_level
from ...frame import Frame
from ....pose import GhostElement, GhostFeature, GhostStateValue, PlayheadOffset, ghost_state

# The blue lamp trails the white by a quarter-turn, so the blue flash fires when the playhead is 0.25 of
# a turn *past* the passive ghost (PlayheadOffset ≈ −0.25·2π, i.e. departing).
_PHASE_OFFSET: float = 0.25


class HauntedFlashSettings(LayerSettings):
    white:  Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="White flash brightness (live players + active ghosts)", newline=True)
    blue:   Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="Blue flash brightness (verified passive ghosts)")
    width:  Field[float] = Field(30.0, min=0.0, max=360.0, step=1.0, description="Flash window width (deg)", newline=True)
    ghosts: Field[bool]  = Field(True, description="Enable ghost flashes", newline=True)


class HauntedFlash(BaseLayer):
    """White flash as the playhead crosses each live player (full ``white``) and each **active** ghost
    (dimmed by Fade), plus a blue flash a quarter-turn later for each **verified** passive ghost (Dwell
    & Motion both 1). Ghost flashes are gated by ``ghosts``."""

    def __init__(self, resolution: int, config: HauntedFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        P = self._config
        centre: float = _PHASE_OFFSET * math.tau   # blue fires 0.25 turn past the passive ghost
        half_rad: float = math.radians(P.width / 2.0)

        tracklets = self._board.get_tracklets()
        flash_white: float = 0.0
        flash_blue:  float = 0.0

        # WHITE — live players flash at the crossing (live poses are fully present; no fade).
        for pose in self._board.get_frames(self._pose_stage).values():
            tracklet = tracklets.get(pose.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            level = offset_to_level(pose[PlayheadOffset].value, half_rad)
            if level > 0.0:
                flash_white = max(flash_white, level * P.white)

        # Ghosts — ACTIVE flash white at their fixed crossing (dimmed by Fade); a VERIFIED passive
        # (dwell & motion both 1) flashes blue a quarter-turn later. A still-building passive gets none.
        if P.ghosts:
            for ghost in self._board.get_ghosts().values():
                state = ghost_state(ghost)
                offset = ghost[PlayheadOffset].value
                if state is GhostStateValue.ACTIVE:
                    level = offset_to_level(offset, half_rad)
                    if level > 0.0:
                        fade = ghost[GhostFeature].get(GhostElement.Fade, 1.0)
                        flash_white = max(flash_white, level * P.white * fade)
                elif state is GhostStateValue.PASSIVE:
                    stab = ghost[GhostFeature]
                    if stab.get(GhostElement.Dwell, 0.0) >= 1.0 and stab.get(GhostElement.Motion, 0.0) >= 1.0:
                        d = offset + centre   # verified → full blue, shifted a quarter-turn then wrapped to [-π, π)
                        level = offset_to_level(math.atan2(math.sin(d), math.cos(d)), half_rad)
                        if level > 0.0:
                            flash_blue = max(flash_blue, level * P.blue)

        half = self.resolution // 2
        if flash_white > 0.0:
            white[:half] += flash_white
        if flash_blue > 0.0:
            blue[:half] += flash_blue
