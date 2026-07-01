"""Ghoster — spawns and manages frozen "ghost" persons from held poses.

When a live person's ``PlayheadStability`` crosses ``threshold`` on a rising edge (with
``release`` hysteresis), Ghoster snapshots their pose into a **ghost**: a frozen ``Frame``
``reidentify``-ed under a pool track id, fixed at the lock azimuth. Each ghost **owns an
azimuth band** (``band_degrees`` wide) that it **claims**. A live person standing inside any
claimed band is **ghosted** — Ghoster stamps ``GhostedFeature`` on them and drops them from
the OSC dict, so the frozen ghost is the only voice there.

Ghoster sits between the LERP pipeline and its fan-out. ``process(frames)`` runs once per tick
and emits on three channels:

* **frames**  → board live store + window tracker: the *tagged live* dict (GhostedFeature stamped).
* **ghosts**  → board ghost store: a snapshot of the authoritative registry.
* **sound**   → OSC: tagged live **minus** ghosted ids, **plus** the ghosts.

The internal ``_ghosts`` registry is the source of truth; the board ghost store is a published
snapshot. Ghosts persist until ``reset``; locks accumulate up to the pool size (then recycle
the oldest or ignore, per ``recycle_oldest``).
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

from .ghosted_feature import GhostedFeature
from .playhead_offset import PlayheadOffset
from .playhead_stability import PlayheadElement, PlayheadStability

logger = logging.getLogger(__name__)


class GhosterSettings(BaseSettings):
    """Configuration for ``Ghoster``."""
    live_players:   Field[int]   = Field(4, access=Field.INIT, visible=False, description="Live player count (shared from root num_players)")
    num_virtual:    Field[int]   = Field(8, access=Field.INIT, visible=False, description="Ghost pool size (shared from root num_virtual)")
    enabled:        Field[bool]  = Field(True,  description="Spawn ghosts and mute live players on claimed spots")
    threshold:      Field[float] = Field(0.95, min=0.0, max=1.0, step=0.01, description="Stability rising-edge level that locks a ghost")
    release:        Field[float] = Field(0.6,  min=0.0, max=1.0, step=0.01, description="Stability must fall below this to re-arm (hysteresis)")
    band_degrees:   Field[float] = Field(10.0, min=0.0, max=180.0, step=0.5, description="Azimuth band a ghost owns (total degrees)")
    recycle_oldest: Field[bool]  = Field(True, description="When the pool is full, recycle the oldest ghost (else ignore new locks)", newline=True)
    reset:          Field[bool]  = Field(False, widget=Widget.button, description="Clear all ghosts")
    num_ghosts:     Field[int]   = Field(0, access=Field.READ, description="Current number of active ghosts", newline=True)


class Ghoster:
    """Lock-detector + ghost registry; see the module docstring."""

    def __init__(self, settings: GhosterSettings, playhead: Callable[[], float]) -> None:
        self._settings = settings
        self._playhead = playhead   # live playhead (radians) — refreshes each ghost's PlayheadOffset
        live = settings.live_players
        self._ghost_ids: list[int] = list(range(live, live + settings.num_virtual))

        self._lock = Lock()
        self._armed: dict[int, bool] = {}       # live id -> may fire (edge + hysteresis)
        self._ghosts: dict[int, Frame] = {}     # ghost id -> frozen frame (source of truth)
        self._ghost_az: dict[int, float] = {}   # ghost id -> lock azimuth (radians)
        self._order: deque[int] = deque()       # ghost ids in creation order (FIFO recycle)

        self._frame_callbacks: list[FrameDictCallback] = []
        self._ghost_callbacks: list[FrameDictCallback] = []
        self._sound_callbacks: list[FrameDictCallback] = []

        self._settings.bind(GhosterSettings.reset, self._on_reset)  # type: ignore[arg-type]

    # -- output channels -----------------------------------------------------

    def add_frames_callback(self, cb: FrameDictCallback) -> None:
        """Tagged-live dict → board live store + window tracker."""
        self._frame_callbacks.append(cb)

    def add_ghosts_callback(self, cb: FrameDictCallback) -> None:
        """Ghost snapshot → board ghost store."""
        self._ghost_callbacks.append(cb)

    def add_sound_callback(self, cb: FrameDictCallback) -> None:
        """Muted-live + ghosts dict → OSC sound sender."""
        self._sound_callbacks.append(cb)

    # -- lifecycle -----------------------------------------------------------

    def stop(self) -> None:
        """Teardown — unbind the reset button (no thread of its own)."""
        self._settings.unbind(GhosterSettings.reset, self._on_reset)  # type: ignore[arg-type]

    def clear(self) -> None:
        """Drop all ghosts and re-arm every live person."""
        with self._lock:
            self._ghosts.clear()
            self._ghost_az.clear()
            self._order.clear()
            for tid in self._armed:
                self._armed[tid] = True
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
            self._detect_locks(frames)
            ghosts = self._live_ghosts(playhead)
            tagged, sound = self._build_outputs(frames, ghosts)
        self._settings.num_ghosts = len(ghosts)
        self._emit(tagged, ghosts, sound)

    def _emit(self, tagged: FrameDict, ghosts: FrameDict, sound: FrameDict) -> None:
        for cb in self._frame_callbacks:
            cb(tagged)
        for cb in self._ghost_callbacks:
            cb(ghosts)
        for cb in self._sound_callbacks:
            cb(sound)

    # -- lock detection (call under _lock) -----------------------------------

    def _detect_locks(self, frames: FrameDict) -> None:
        thr = self._settings.threshold
        rel = self._settings.release
        for tid, frame in frames.items():
            s = frame[PlayheadStability].get(PlayheadElement.Stability)
            if math.isnan(s):
                s = 0.0
            az = frame[Azimuth].value
            armed = self._armed.get(tid, True)
            if armed and s >= thr and not math.isnan(az) and self._band_owner(az) is None:
                self._spawn_ghost(frame, az)
                self._armed[tid] = False
            elif s < rel:
                self._armed[tid] = True
            else:
                self._armed[tid] = armed
        # Re-arm live ids that are no longer tracked.
        for tid in self._armed:
            if tid not in frames:
                self._armed[tid] = True

    def _spawn_ghost(self, frame: Frame, az: float) -> None:
        gid = self._free_ghost_id()
        if gid is None:
            return  # pool full and recycling disabled — ignore this lock
        self._ghosts[gid] = reidentify(frame, gid)
        self._ghost_az[gid] = az
        if gid in self._order:
            self._order.remove(gid)
        self._order.append(gid)

    def _free_ghost_id(self) -> int | None:
        for gid in self._ghost_ids:
            if gid not in self._ghosts:
                return gid
        if self._settings.recycle_oldest and self._order:
            return self._order[0]   # oldest — _spawn_ghost overwrites + re-orders it
        return None

    def _band_owner(self, az: float) -> int | None:
        """Ghost id whose band contains ``az`` (radians), or ``None``. Call under ``_lock``."""
        half = math.radians(self._settings.band_degrees / 2.0)
        for gid, gaz in self._ghost_az.items():
            d = az - gaz
            if abs(math.atan2(math.sin(d), math.cos(d))) <= half:
                return gid
        return None

    def _live_ghosts(self, playhead: float) -> FrameDict:
        """The frozen ghost registry with each ghost's ``PlayheadOffset`` recomputed from its fixed
        azimuth and the current ``playhead`` — the pose stays frozen, only the playhead-relative
        offset sweeps. Call under ``_lock``. A NaN playhead (no meaningful playhead) leaves the
        frozen offsets untouched, matching ``PlayheadOffsetExtractor``."""
        if math.isnan(playhead):
            return dict(self._ghosts)
        return {gid: replace(g, {PlayheadOffset: PlayheadOffset.from_value(self._ghost_az[gid] - playhead)})
                for gid, g in self._ghosts.items()}

    # -- output build (call under _lock) -------------------------------------

    def _build_outputs(self, frames: FrameDict, ghosts: FrameDict) -> tuple[FrameDict, FrameDict]:
        tagged: FrameDict = {}
        sound: FrameDict = {}
        for tid, frame in frames.items():
            az = frame[Azimuth].value
            ghosted = not math.isnan(az) and self._band_owner(az) is not None
            tagged_frame = replace(frame, {GhostedFeature: GhostedFeature.from_value(1.0 if ghosted else 0.0, 1.0)})
            tagged[tid] = tagged_frame
            if not ghosted:
                sound[tid] = tagged_frame
        sound.update(ghosts)   # ghost ids never collide with live ids
        return tagged, sound

    # -- settings ------------------------------------------------------------

    def _on_reset(self, _=None) -> None:
        self.clear()
        logger.info("Ghoster: cleared all ghosts")
