"""BeamHaunted — the beam-mode player/ghost flash (a debug/experimentation layer,
never in a state's mix; pairs with ``pose.ghoster.enabled`` for solo ghost sessions).

WHITE (front-lamp) channel: a flash as the rotating playhead crosses each live player (at full
``white``), and — when ``ghosts`` is enabled — as it crosses each **active** ghost at its fixed
azimuth, dimmed by that ghost's ``GhostFeature`` Fade (a released ghost decays 1→0).

BLUE (left-lamp) channel: a flash for each **verified** passive ghost — one whose ``Dwell`` and
``Motion`` have both reached ``1.0`` (settled = ready to break free into an active ghost) — fired a
fixed quarter-turn *later* than the crossing (``PlayheadOffset ≈ −0.25·2π``, the departing side),
so it trails the white. A still-building passive ghost gets no blue. Fade is irrelevant to blue.

A single ``width`` sizes every window; ``white`` / ``blue`` set the two flash brightnesses and
``base_white`` a constant front-lamp floor the white flash rides on. Dwell/Motion no longer shape
the flash. When the sweep steps clean over a narrow window (fast crossings — e.g. a person just
repositioned), ``_closest_pass`` still guarantees one flash on the frame nearest the pose. This
module owns both window kernels (``offset_to_level``, ``_closest_pass``). No gap.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BeamLayer, LayerSettings
from ...frame import Frame
from ....pose import GhostElement, GhostFeature, GhostStateValue, PlayheadOffset, ghost_state

# The blue lamp trails the white by a quarter-turn, so the blue flash fires when the playhead is 0.25 of
# a turn *past* the passive ghost (PlayheadOffset ≈ −0.25·2π, i.e. departing).
_PHASE_OFFSET: float = 0.25

# Only guarantee a flash on the near half of the sweep; the far side (|offset| → π) never triggers.
_HALF_PI: float = math.pi / 2.0


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


def _closest_pass(prev: float, cur: float) -> bool:
    """True on the sample where the playhead is nearest the pose — the local minimum of ``|offset|``
    as the sweep passes it. A constant-velocity one-step prediction lets it fire in real time on the
    closest frame (which may sit just *before* or just *after* zero), not a frame late. Guarantees at
    least one flash per pass even when the ``width`` window is too narrow for any sample to land in it.

    Gated to the near half so the far side never fires; NaN (no prev yet / motor stopped) never fires.
    Steps are small (~7°) and the gate keeps ``cur`` off the ±π wrap, so a plain difference is a safe
    velocity estimate here.
    """
    if math.isnan(prev) or math.isnan(cur) or abs(cur) >= _HALF_PI:
        return False
    nxt = cur + (cur - prev)                 # predicted next offset (constant velocity)
    return abs(cur) <= abs(prev) and abs(cur) <= abs(nxt)


class BeamHauntedSettings(LayerSettings):
    white:      Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="White flash brightness (live players + active ghosts)")
    base_white: Field[float] = Field(0.0, min=0.0, max=1.0,    step=0.01, description="White base brightness of the front lamps when not flashing")
    blue:       Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="Blue flash brightness (verified passive ghosts)")
    width:      Field[float] = Field(30.0, min=0.0, max=360.0, step=1.0, description="Flash window width (deg)")
    ghosts:     Field[bool]  = Field(True, description="Enable ghost flashes")


class BeamHaunted(BeamLayer):
    """White flash as the playhead crosses each live player (full ``white``) and each **active** ghost
    (dimmed by Fade), plus a blue flash a quarter-turn later for each **verified** passive ghost (Dwell
    & Motion both 1). Ghost flashes are gated by ``ghosts``."""

    def __init__(self, resolution: int, config: BeamHauntedSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage
        # Previous-tick offset per source, for the closest-approach guarantee. Rebuilt each _draw so
        # vanished ids are pruned (a re-appeared id can't carry a stale prev across a gap).
        self._prev_live:  dict[int, float] = {}   # live players — raw offset
        self._prev_ghost: dict[int, float] = {}   # active ghosts — raw offset
        self._prev_blue:  dict[int, float] = {}   # passive ghosts — quarter-turn-shifted offset

    def reset(self) -> None:
        self._prev_live.clear()
        self._prev_ghost.clear()
        self._prev_blue.clear()

    def _draw(self, frame: Frame, beam_lights: np.ndarray) -> None:
        P = self._config
        centre: float = _PHASE_OFFSET * math.tau   # blue fires 0.25 turn past the passive ghost
        half_rad: float = math.radians(P.width / 2.0)

        flash_white: float = 0.0
        flash_blue:  float = 0.0
        prev_live:  dict[int, float] = {}
        prev_ghost: dict[int, float] = {}
        prev_blue:  dict[int, float] = {}

        # WHITE — live players flash at the crossing (live poses are fully present; no fade). The width
        # window lights the near frames; _closest_pass guarantees the nearest one if the window is skipped.
        for pose in self._board.get_frames(self._pose_stage).values():
            offset = pose[PlayheadOffset].value
            level = offset_to_level(offset, half_rad)
            if _closest_pass(self._prev_live.get(pose.track_id, math.nan), offset):
                level = 1.0
            prev_live[pose.track_id] = offset
            if level > 0.0:
                flash_white = max(flash_white, level * P.white)

        # Ghosts — ACTIVE flash white at their fixed crossing (dimmed by Fade); a VERIFIED passive
        # (dwell & motion both 1) flashes blue a quarter-turn later. A still-building passive gets none.
        if P.ghosts:
            for gid, ghost in self._board.get_ghosts().items():
                state = ghost_state(ghost)
                offset = ghost[PlayheadOffset].value
                if state is GhostStateValue.ACTIVE:
                    level = offset_to_level(offset, half_rad)
                    if _closest_pass(self._prev_ghost.get(gid, math.nan), offset):
                        level = 1.0
                    prev_ghost[gid] = offset
                    if level > 0.0:
                        fade = ghost[GhostFeature].get(GhostElement.Fade, 1.0)
                        flash_white = max(flash_white, level * P.white * fade)
                elif state is GhostStateValue.PASSIVE:
                    d = math.atan2(math.sin(offset + centre), math.cos(offset + centre))   # shift 0.25 turn, wrap to [-π, π)
                    # Track prev every tick (even while unverified) so the velocity stays continuous.
                    prev_d = self._prev_blue.get(gid, math.nan)
                    prev_blue[gid] = d
                    stab = ghost[GhostFeature]
                    if stab.get(GhostElement.Dwell, 0.0) >= 1.0 and stab.get(GhostElement.Motion, 0.0) >= 1.0:
                        level = offset_to_level(d, half_rad)
                        if _closest_pass(prev_d, d):
                            level = 1.0
                        if level > 0.0:
                            flash_blue = max(flash_blue, level * P.blue)

        self._prev_live, self._prev_ghost, self._prev_blue = prev_live, prev_ghost, prev_blue

        # Constant front-lamp floor brightened by the white flash; the blue flash on the left lamp.
        self._add_beam_lights(beam_lights, front_white=P.base_white + flash_white, left_blue=flash_blue)
