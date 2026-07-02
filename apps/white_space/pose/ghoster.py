"""Ghoster — leaves held poses behind as frozen "ghost" persons, sampled per playhead sweep.

Each time the rotating playhead sweeps across a person (a ``PlayheadOffset`` zero-crossing near 0),
Ghoster compares his azimuth to where he was at the **previous** sweep. If he moved more than
``band_degrees`` since then — and the pose he left had **settled** on the spot (``PlayheadStability``
Dwell and Motion both 1) — the pose he held at that previous spot is **committed** as a ghost there.
A ghost is deleted only as the playhead **sweeps across it** while a live pose overlaps it (within
``band_degrees``) — a live person reclaims the spot; a just-made ghost is skipped on its birth tick, so
you can't wipe a ghost the instant you leave it. A ghost is a frozen ``Frame`` ``reidentify``-ed under a
pool track id, fixed at the commit azimuth; its ``PlayheadOffset`` is refreshed each tick so it sweeps
with the playhead. Each ghost also carries a ``Fade`` (1.0 when left behind) that steps down by
``1/fade_sweeps`` on each sweep where it is not reclaimed, and is removed at 0 — so an unreclaimed ghost
fades out over ``fade_sweeps`` sweeps.

There is no muting: every live person is always audible; ghosts are simply extra OSC voices.

Ghoster sits between the LERP pipeline and its fan-out. ``process(frames)`` runs once per tick and
emits on three channels:

* **frames**  → board live store + window tracker: the live frames, unchanged.
* **ghosts**  → board ghost store: the ghost registry (``PlayheadOffset`` refreshed).
* **sound**   → OSC: live frames **plus** the ghosts.

The internal ``_ghosts`` registry is the source of truth; the board ghost store is a published
snapshot. Ghosts persist until ``reset``; commits accumulate up to the pool size (then recycle the
oldest or ignore, per ``recycle_oldest``).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from threading import Lock
from typing import Callable

from modules.pose.frame import Frame, FrameDict, FrameDictCallback, replace, reidentify
from modules.pose.features import Azimuth
from modules.settings import BaseSettings, Field, Widget

from .fade import Fade
from .playhead_offset import PlayheadOffset
from .playhead_stability import PlayheadElement, PlayheadStability

logger = logging.getLogger(__name__)

# Only treat a PlayheadOffset sign change as a sweep hit when both samples are within a quarter-turn
# of the playhead, so the ±π wrap (opposite side of the ring) never counts.
_HALF_PI: float = math.pi / 2.0


class GhosterSettings(BaseSettings):
    """Configuration for ``Ghoster``."""
    live_players:   Field[int]   = Field(4, access=Field.INIT, visible=False, description="Live player count (shared from root num_players)")
    num_virtual:    Field[int]   = Field(8, access=Field.INIT, visible=False, description="Ghost pool size (shared from root num_virtual)")
    enabled:        Field[bool]  = Field(True,  description="Record and commit ghosts")
    band_degrees:   Field[float] = Field(10.0, min=0.0, max=180.0, step=0.5, description="Distance (deg) that counts as the same spot / must be exceeded to leave a ghost")
    fade_sweeps:    Field[int]   = Field(16, min=1, max=256, step=1, description="Playhead sweeps over which a ghost fades out, then is removed")
    recycle_oldest: Field[bool]  = Field(True, description="When the pool is full, recycle the oldest ghost (else ignore new commits)", newline=True)
    reset:          Field[bool]  = Field(False, widget=Widget.button, description="Clear all ghosts")
    num_ghosts:     Field[int]   = Field(0, access=Field.READ, description="Current number of active ghosts", newline=True)


class Ghoster:
    """Record-and-commit ghost registry; see the module docstring."""

    def __init__(self, settings: GhosterSettings, playhead: Callable[[], float]) -> None:
        self._settings = settings
        self._playhead = playhead   # live playhead (radians) — refreshes each ghost's PlayheadOffset
        live = settings.live_players
        self._ghost_ids: list[int] = list(range(live, live + settings.num_virtual))

        self._lock = Lock()
        self._prev_offset: dict[int, float] = {}  # live id -> last PlayheadOffset (sweep-crossing detection)
        self._last_az: dict[int, float] = {}      # live id -> azimuth at the previous playhead sweep
        self._last_frame: dict[int, Frame] = {}   # live id -> pose at the previous playhead sweep
        self._ghosts: dict[int, Frame] = {}      # ghost id -> frozen frame (source of truth)
        self._ghost_az: dict[int, float] = {}    # ghost id -> commit azimuth (radians)
        self._order: deque[int] = deque()        # ghost ids in commit order (FIFO recycle)
        self._ghost_fade: dict[int, float] = {}         # ghost id -> presence in [0, 1] (1.0 at commit)
        self._ghost_prev_offset: dict[int, float] = {}  # ghost id -> last offset (sweep-crossing detection)
        self._ghost_maker: dict[int, int] = {}          # ghost id -> live id that made it (can't reclaim its own)

        self._frame_callbacks: list[FrameDictCallback] = []
        self._ghost_callbacks: list[FrameDictCallback] = []
        self._sound_callbacks: list[FrameDictCallback] = []

        self._settings.bind(GhosterSettings.reset, self._on_reset)  # type: ignore[arg-type]

    # -- output channels -----------------------------------------------------

    def add_frames_callback(self, cb: FrameDictCallback) -> None:
        """Live frames → board live store + window tracker."""
        self._frame_callbacks.append(cb)

    def add_ghosts_callback(self, cb: FrameDictCallback) -> None:
        """Ghost snapshot → board ghost store."""
        self._ghost_callbacks.append(cb)

    def add_sound_callback(self, cb: FrameDictCallback) -> None:
        """Live + ghosts dict → OSC sound sender."""
        self._sound_callbacks.append(cb)

    # -- lifecycle -----------------------------------------------------------

    def stop(self) -> None:
        """Teardown — unbind the reset button (no thread of its own)."""
        self._settings.unbind(GhosterSettings.reset, self._on_reset)  # type: ignore[arg-type]

    def clear(self) -> None:
        """Drop all ghosts and every in-progress recording."""
        with self._lock:
            self._ghosts.clear()
            self._ghost_az.clear()
            self._order.clear()
            self._ghost_fade.clear()
            self._ghost_prev_offset.clear()
            self._ghost_maker.clear()
            self._prev_offset.clear()
            self._last_az.clear()
            self._last_frame.clear()
        self._settings.num_ghosts = 0

    # -- main transform ------------------------------------------------------

    def process(self, frames: FrameDict) -> None:
        if not self._settings.enabled:
            # Bypass: pass live frames through untouched, publish no ghosts.
            with self._lock:
                count = len(self._ghosts)
            self._settings.num_ghosts = count
            self._emit(frames, {}, frames)
            return

        playhead = self._playhead()   # read once, outside the lock
        with self._lock:
            self._sweep_sample(frames)            # once per sweep: leave a ghost if he moved > band
            self._advance_ghosts(playhead, frames)  # per ghost sweep: reclaim if overlapped, else fade
            ghosts = self._live_ghosts(playhead)
            sound = {**frames, **ghosts}   # ghost ids never collide with live ids; no muting
        self._settings.num_ghosts = len(ghosts)
        self._emit(frames, ghosts, sound)

    def _emit(self, tagged: FrameDict, ghosts: FrameDict, sound: FrameDict) -> None:
        for cb in self._frame_callbacks:
            cb(tagged)
        for cb in self._ghost_callbacks:
            cb(ghosts)
        for cb in self._sound_callbacks:
            cb(sound)

    # -- sweep sample, delete & commit (call under _lock) --------------------

    def _sweep_sample(self, frames: FrameDict) -> None:
        """Once per playhead sweep (a PlayheadOffset zero-crossing near 0), compare each person's
        azimuth to where he was at the previous sweep; if he moved > ``band_degrees``, leave a ghost
        at that previous spot with the pose he held there. Call under ``_lock``."""
        band = math.radians(self._settings.band_degrees)
        for tid, frame in frames.items():
            offset = frame[PlayheadOffset].value
            prev = self._prev_offset.get(tid, float("nan"))
            self._prev_offset[tid] = offset      # updated every tick to catch the crossing
            hit = (not math.isnan(prev) and prev * offset < 0.0
                   and abs(prev) < _HALF_PI and abs(offset) < _HALF_PI)
            if not hit:
                continue
            az = frame[Azimuth].value
            if math.isnan(az):
                continue
            last_az = self._last_az.get(tid)
            if (last_az is not None
                    and abs(math.atan2(math.sin(az - last_az), math.cos(az - last_az))) > band
                    and self._settled(self._last_frame[tid])):
                self._commit(self._last_frame[tid], last_az)   # moved on from a settled spot → leave a ghost
            self._last_az[tid] = az
            self._last_frame[tid] = frame
        # Drop per-person state for vanished tracks (vanishing leaves nothing).
        present = set(frames)
        for store in (self._prev_offset, self._last_az, self._last_frame):
            for tid in [t for t in store if t not in present]:
                del store[tid]

    @staticmethod
    def _settled(frame: Frame) -> bool:
        """Has the pose settled on its spot — ``PlayheadStability`` Dwell and Motion both full? Only
        then is the pose worth leaving behind (a quick pass-through never fills dwell)."""
        stab = frame[PlayheadStability]
        return stab.get(PlayheadElement.Dwell, 0.0) >= 1.0 and stab.get(PlayheadElement.Motion, 0.0) >= 1.0

    def _advance_ghosts(self, playhead: float, frames: FrameDict) -> None:
        """Each time the playhead sweeps across a ghost (its offset crosses zero): if a live pose
        overlaps it, **remove it** (a live person reclaims the spot); otherwise fade it by
        ``1/fade_sweeps`` and remove it once faded out. A just-made ghost has a NaN prev-offset, so it
        registers no crossing on its birth tick and can't be reclaimed until its next real sweep.
        Frozen while the playhead is NaN. Call under ``_lock``."""
        if math.isnan(playhead):
            return
        band = math.radians(self._settings.band_degrees)
        step = 1.0 / max(1, self._settings.fade_sweeps)
        gone: list[int] = []
        for gid, gaz in self._ghost_az.items():
            offset = math.atan2(math.sin(gaz - playhead), math.cos(gaz - playhead))
            prev = self._ghost_prev_offset[gid]
            self._ghost_prev_offset[gid] = offset
            swept = (not math.isnan(prev) and prev * offset < 0.0
                     and abs(prev) < _HALF_PI and abs(offset) < _HALF_PI)
            if not swept:
                continue
            if self._person_overlaps(gaz, frames, band):
                gone.append(gid)                       # a live pose overlaps → reclaim the spot
            else:
                self._ghost_fade[gid] -= step
                if self._ghost_fade[gid] <= 0.0:
                    gone.append(gid)
        for gid in gone:
            self._remove_ghost(gid)

    def _person_overlaps(self, gaz: float, frames: FrameDict, band: float) -> bool:
        """Is any live person's azimuth within ``band`` of ``gaz``? Call under ``_lock``."""
        for frame in frames.values():
            az = frame[Azimuth].value
            if not math.isnan(az) and abs(math.atan2(math.sin(az - gaz), math.cos(az - gaz))) <= band:
                return True
        return False

    def _remove_ghost(self, gid: int) -> None:
        """Drop a ghost and all its per-ghost state. Call under ``_lock``."""
        self._ghosts.pop(gid, None)
        self._ghost_az.pop(gid, None)
        self._ghost_fade.pop(gid, None)
        self._ghost_prev_offset.pop(gid, None)
        if gid in self._order:
            self._order.remove(gid)

    def _commit(self, frame: Frame, az: float) -> None:
        gid = self._band_owner(az)          # existing ghost within band → replace it (override)
        if gid is None:
            gid = self._free_ghost_id()     # else a new / recycled pool id
            if gid is None:
                return                       # pool full and recycling disabled — ignore
        self._set_ghost(gid, frame, az)

    def _set_ghost(self, gid: int, frame: Frame, az: float) -> None:
        # Pin the ghost to the anchor spot, overriding the frame's own (slightly drifted) azimuth.
        self._ghosts[gid] = replace(reidentify(frame, gid), {Azimuth: Azimuth.from_value(az)})
        self._ghost_az[gid] = az
        self._ghost_fade[gid] = 1.0             # fresh / replaced ghost starts fully present
        self._ghost_prev_offset[gid] = float("nan")
        if gid in self._order:
            self._order.remove(gid)
        self._order.append(gid)

    def _free_ghost_id(self) -> int | None:
        for gid in self._ghost_ids:
            if gid not in self._ghosts:
                return gid
        if self._settings.recycle_oldest and self._order:
            return self._order[0]   # oldest — _set_ghost overwrites + re-orders it
        return None

    def _band_owner(self, az: float) -> int | None:
        """Ghost id within ``band_degrees`` (radius) of ``az``, or ``None``. Call under ``_lock``."""
        band = math.radians(self._settings.band_degrees)
        for gid, gaz in self._ghost_az.items():
            d = az - gaz
            if abs(math.atan2(math.sin(d), math.cos(d))) <= band:
                return gid
        return None

    def _live_ghosts(self, playhead: float) -> FrameDict:
        """The frozen ghost registry, each ghost stamped with its current ``Fade`` and — when the
        playhead is finite — its ``PlayheadOffset`` recomputed from its fixed azimuth and the current
        ``playhead`` (the pose stays frozen; only fade and the playhead-relative offset change). Call
        under ``_lock``."""
        out: FrameDict = {}
        for gid, g in self._ghosts.items():
            feats: dict = {Fade: Fade.from_value(self._ghost_fade[gid])}
            if not math.isnan(playhead):
                feats[PlayheadOffset] = PlayheadOffset.from_value(self._ghost_az[gid] - playhead)
            out[gid] = replace(g, feats)
        return out

    # -- settings ------------------------------------------------------------

    def _on_reset(self, _=None) -> None:
        self.clear()
        logger.info("Ghoster: cleared all ghosts")
