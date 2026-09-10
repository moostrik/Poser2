"""PlayheadFlash — the INTRO flash: the beam lights strike on while the rotating playhead
crosses each player, driven by the continuous ``PlayheadOffset``.

Each pose's signed offset to the playhead defines an on/off window around the crossing: the
flash switches on while the playhead is within the ``width``° window (±``width``/2 each side),
save a dark ``gap`` notch straddling the crossing itself. Every hit flashes the same — the
brightness is plain ``white`` / ``blue``, one level for everyone. The ghost-driven variant,
where active ghosts flash dimmed by their Fade and verified passive ghosts flash blue a
quarter-turn late, is the separate ``playhead_haunted`` debug layer.

When the sweep steps clean over a narrow window (a fast crossing, or a person who just
repositioned), ``_closest_pass`` still guarantees one flash on the frame nearest the pose —
so a hit is never silently skipped. It is bypassed when a ``gap`` notch is configured, since
that asks for darkness at exactly the crossing the guarantee would fire on.

This module owns both flash kernels (``offset_to_level``, ``_closest_pass``);
``playhead_haunted`` imports them.
"""

import math

import numpy as np

from modules.settings import Field

from .._base_layer import BeamLayer, LayerSettings
from ...frame import Frame
from ....pose import PlayheadOffset

# Only guarantee a flash on the near half of the ring; the far side (|offset| → π) never triggers.
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


class PlayheadFlashSettings(LayerSettings):
    base_white: Field[float] = Field(0.0, min=0.0, max=1.0,    step=0.01, description="Base brightness of the front white lamp")
    base_blue:  Field[float] = Field(0.0, min=0.0, max=1.0,    step=0.01, description="Base brightness of both blue lamps")
    white:      Field[float] = Field(1.0, min=0.0, max=1.0,    step=0.01, description="White flash brightness", newline=True)
    blue:       Field[float] = Field(0.0, min=0.0, max=1.0,    step=0.01, description="Blue flash brightness")
    width:      Field[float] = Field(20.0, min=0.0, max=360.0, step=1.0, description="Flash window width (deg) — 0 still flashes one frame per pass (the closest-pass guarantee)", newline=True)
    gap:        Field[float] = Field(0.0, min=0.0, max=1.0,    step=0.01, description="Fraction of the window centre that stays dark — a notch at the crossing (0 = solid; any notch disables the closest-pass guarantee)")


class PlayheadFlash(BeamLayer):
    """Continuous base level plus an on/off flash window tracking the playhead's approach to
    each active player, read from ``PlayheadOffset``; see the module docstring."""

    def __init__(self, resolution: int, config: PlayheadFlashSettings, board, pose_stage: int) -> None:
        super().__init__(resolution, config, board)
        self._config = config
        self._pose_stage = pose_stage
        # Previous-tick offset per pose, for the closest-approach guarantee. Rebuilt each _draw so
        # vanished ids are pruned (a re-appeared id can't carry a stale prev across a gap).
        self._prev_offsets: dict[int, float] = {}

    def reset(self) -> None:
        self._prev_offsets.clear()

    def _draw(self, frame: Frame, beam_lights: np.ndarray) -> None:
        P = self._config
        half_rad: float = math.radians(P.width / 2.0)
        guarantee: bool = P.gap <= 0.0     # a notch asks for darkness exactly where this would fire

        tracklets = self._board.get_tracklets()
        flash_white: float = 0.0
        flash_blue:  float = 0.0
        prev_offsets: dict[int, float] = {}
        for pose in self._board.get_frames(self._pose_stage).values():
            tracklet = tracklets.get(pose.track_id)
            if tracklet is None or not tracklet.is_active:
                continue
            offset = pose[PlayheadOffset].value
            level = offset_to_level(offset, half_rad, P.gap)
            if guarantee and _closest_pass(self._prev_offsets.get(pose.track_id, math.nan), offset):
                level = 1.0
            prev_offsets[pose.track_id] = offset
            if level <= 0.0:
                continue
            flash_white = max(flash_white, level * P.white)
            flash_blue  = max(flash_blue,  level * P.blue)
        self._prev_offsets = prev_offsets

        # The flash is the front white lamp plus both blue lamps, on a constant base.
        self._add_beam_lights(beam_lights,
                             front_white=P.base_white + flash_white,
                             left_blue=P.base_blue + flash_blue,
                             right_blue=P.base_blue + flash_blue)
