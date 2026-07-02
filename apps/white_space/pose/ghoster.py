"""Ghoster — active & passive "ghost" persons left behind from intentional held poses.

Each live person carries a **passive ghost**: a candidate that follows him sweep-by-sweep (a
``PlayheadOffset`` zero-crossing marks each sweep) while he stays within ``band_degrees`` of its spot,
accumulating the ghost's own **dwell** (sweeps on the spot), **motion** (on-spot ``MotionTime``), and
**stability** — how many consecutive sweeps his pose has stayed similar (a Gaussian over ``Angles``). A
pose held for ``stability_sweeps`` sweeps becomes **valid**: the Ghoster remembers that pose and the spot
it was held at, replacing it whenever a newer pose is held. When he **breaks free** — the moment he moves
> ``band_degrees`` from his last-swept spot (checked every tick, not only on a sweep) — and the passive
ghost had **settled** (dwell & motion both full, and a valid pose held no more than ``stability_max_age``
sweeps ago), the **last valid held pose is frozen at the spot he held it** as an **active ghost** — never
the transitional pose/position he is in while walking away; otherwise it is dropped. A fresh passive
re-anchors at his new spot on the next sweep. Slow drift never breaks free, so it leaves nothing.

Ghosts are ordinary pose ``Frame``s carrying a ``GhostState`` feature (``PASSIVE`` / ``ACTIVE``); both
kinds live in the one board ghost store, told apart by that feature. Only **active** ghosts are sent to
OSC. An active ghost is a frozen ``reidentify``-ed frame on a pool id, its ``PlayheadOffset`` refreshed
each tick so it sweeps with the playhead; it carries a ``Fade`` that steps down by ``1/fade_sweeps`` on
each sweep it is not reclaimed (removed at 0), and is **reclaimed** (removed) when the playhead sweeps it
while a live pose overlaps it — a just-made ghost is skipped on its birth tick.

The Ghoster owns the per-pose metrics: it stamps a ``GhostFeature`` (**Dwell**, **Motion**, **Stability**,
**Fade**, each in [0, 1]) on every frame it emits — live poses and ghosts — so the GPU renderer / light /
OSC read the same band-based values that drive the gate. A live pose has ``Fade`` 1.0; an active ghost's
``Fade`` decays. The ghost *kind* rides separately on ``GhostState`` (absent ⇒ live).

There is no muting: every live person is always audible; ghosts are extra OSC voices. ``process(frames)``
runs once per tick and emits on three channels:

* **frames** → board live store + window tracker: live frames tagged with dwell/motion.
* **ghosts** → board ghost store: passive + active ghosts (each carrying ``GhostState``).
* **sound**  → OSC: tagged live frames **plus active ghosts only**.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Callable

import numpy as np

from modules.pose.frame import Frame, FrameDict, FrameDictCallback, replace, reidentify
from modules.pose.features import Angles, Azimuth, MotionTime
from modules.settings import BaseSettings, Field, Widget

from .ghost_feature import GhostFeature
from .ghost_state import GhostState, GhostStateValue
from .playhead_offset import PlayheadOffset

logger = logging.getLogger(__name__)

# Sweep-crossing guard: only treat a PlayheadOffset sign change as a hit when both samples are within a
# quarter-turn of the playhead, so the ±π wrap (opposite side of the ring) never counts.
_HALF_PI: float = math.pi / 2.0
_TWO_PI:  float = 2.0 * math.pi


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _pose_similarity(current: Angles, older: Angles, scale: float) -> float:
    """Confidence-weighted harmonic-mean Gaussian similarity in ``[0, 1]`` between two poses — used to tell
    whether a pose is being *held* from one sweep to the next. Reuses ``Angles.subtract`` for the wrapped
    per-joint difference; the harmonic mean is strict, so one moved joint pulls the whole score down."""
    diff = current.subtract(older)
    valid = diff.scores > 0.0
    if not valid.any():
        return 0.0
    sim = np.exp(-np.square(diff.values[valid] / scale))   # per-joint Gaussian, (0, 1]
    weights = diff.scores[valid]
    return float(weights.sum() / (weights / sim).sum())     # weighted harmonic mean


@dataclass
class _Passive:
    """A live person's candidate ghost: follows him and builds its own dwell/motion/stability."""
    spot: float                  # anchor azimuth (his position at the last sweep)
    frame: Frame                 # pose held at the spot (refreshed each sweep)
    sweeps: int                  # consecutive sweeps on this spot (dwell)
    anchor_motion: float         # MotionTime snapshot at motion_sweeps (NaN until reached)
    stable_sweeps: int           # consecutive sweeps his pose has stayed similar (starts 1)
    stable_frame: Frame | None   # last pose held long enough to be valid (None until then)
    stable_spot: float           # azimuth where that valid pose was held
    stable_age: int              # sweeps since the valid pose was last held (0 while held)


class GhosterSettings(BaseSettings):
    """Configuration for ``Ghoster``."""
    live_players:        Field[int]   = Field(4, access=Field.INIT, visible=False, description="Live player count (shared from root num_players)")
    num_virtual:         Field[int]   = Field(8, access=Field.INIT, visible=False, description="Ghost pool size (shared from root num_virtual)")
    enabled:             Field[bool]  = Field(True, description="Record and commit ghosts")
    reset:               Field[bool]  = Field(False, widget=Widget.button, description="Clear all ghosts")
    recycle_oldest:      Field[bool]  = Field(True, description="When the pool is full, recycle the oldest ghost (else ignore new commits)")
    num_ghosts:          Field[int]   = Field(0, access=Field.READ, description="Current number of active ghosts")
    band_degrees:        Field[float] = Field(10.0, min=0.0, max=180.0, step=0.5, description="Distance (deg) that counts as the same spot / must be exceeded to break free")
    dwell_sweeps:        Field[int]   = Field(4, min=1, max=32, step=1, description="Playhead sweeps on a spot for dwell to fill", newline=True)
    motion_sweeps:       Field[int]   = Field(2, min=1, max=32, step=1, description="Sweeps on a spot before motion starts counting")
    motion_scale:        Field[float] = Field(5.0, min=0.1, max=50.0, step=0.1, description="On-spot MotionTime that maps motion to 1.0")
    stability_sweeps:    Field[int]   = Field(2, min=1, max=32, step=1, description="Sweeps a pose must be held to become a valid ghost pose (1 = off)", newline=True)
    stability_max_age:   Field[int]   = Field(3, min=0, max=32, step=1, description="A valid pose expires this many sweeps after it was last held")
    stability_scale:     Field[float] = Field(0.8, min=0.1, max=2.0, step=0.05, description="Pose-similarity scale (rad) — larger = more forgiving")
    stability_threshold: Field[float] = Field(0.9, min=0.0, max=1.0, step=0.01, description="Min pose similarity between sweeps to count as 'held'")
    fade_sweeps:         Field[int]   = Field(16, min=1, max=256, step=1, description="Playhead sweeps over which an active ghost fades out, then is removed", newline=True)


class Ghoster:
    """Passive/active ghost registry; see the module docstring."""

    def __init__(self, settings: GhosterSettings, playhead: Callable[[], float]) -> None:
        self._settings = settings
        self._playhead = playhead   # live playhead (radians) — refreshes each ghost's PlayheadOffset
        live = settings.live_players
        self._ghost_ids: list[int] = list(range(live, live + settings.num_virtual))

        self._lock = Lock()
        self._prev_offset: dict[int, float] = {}   # live id -> last PlayheadOffset (sweep-crossing detection)
        self._passive: dict[int, _Passive] = {}    # live id -> candidate ghost following him
        self._ghosts: dict[int, Frame] = {}        # active ghost id -> frozen frame (source of truth)
        self._ghost_az: dict[int, float] = {}      # active ghost id -> commit azimuth (radians)
        self._order: deque[int] = deque()          # active ghost ids in commit order (FIFO recycle)
        self._ghost_fade: dict[int, float] = {}         # active ghost id -> presence in [0, 1] (1.0 at commit)
        self._ghost_prev_offset: dict[int, float] = {}  # active ghost id -> last offset (sweep-crossing detection)

        self._frame_callbacks: list[FrameDictCallback] = []
        self._ghost_callbacks: list[FrameDictCallback] = []
        self._sound_callbacks: list[FrameDictCallback] = []

        self._settings.bind(GhosterSettings.reset, self._on_reset)  # type: ignore[arg-type]

    # -- output channels -----------------------------------------------------

    def add_frames_callback(self, cb: FrameDictCallback) -> None:
        """Tagged live frames (dwell/motion stamped) → board live store + window tracker."""
        self._frame_callbacks.append(cb)

    def add_ghosts_callback(self, cb: FrameDictCallback) -> None:
        """Passive + active ghosts (with ``GhostState``) → board ghost store."""
        self._ghost_callbacks.append(cb)

    def add_sound_callback(self, cb: FrameDictCallback) -> None:
        """Tagged live frames + active ghosts → OSC sound sender."""
        self._sound_callbacks.append(cb)

    # -- lifecycle -----------------------------------------------------------

    def stop(self) -> None:
        """Teardown — unbind the reset button (no thread of its own)."""
        self._settings.unbind(GhosterSettings.reset, self._on_reset)  # type: ignore[arg-type]

    def clear(self) -> None:
        """Drop all ghosts (active + passive candidates)."""
        with self._lock:
            self._ghosts.clear()
            self._ghost_az.clear()
            self._order.clear()
            self._ghost_fade.clear()
            self._ghost_prev_offset.clear()
            self._prev_offset.clear()
            self._passive.clear()
        self._settings.num_ghosts = 0

    # -- main transform ------------------------------------------------------

    def process(self, frames: FrameDict) -> None:
        if not self._settings.enabled:
            # Bypass: pass live frames through untouched (no dwell/motion), publish no ghosts.
            with self._lock:
                count = len(self._ghosts)
            self._settings.num_ghosts = count
            self._emit(frames, {}, frames)
            return

        playhead = self._playhead()   # read once, outside the lock
        with self._lock:
            self._sweep_sample(frames)              # update passive ghosts; promote on break-free
            self._advance_ghosts(playhead, frames)  # active ghosts: fade / reclaim on sweep
            tagged = self._tag_live(frames)         # stamp dwell/motion/fade (GhostFeature) on live frames
            active = self._live_ghosts(playhead)    # active ghosts, PlayheadOffset + GhostFeature refreshed
            ghosts = {**self._passive_ghosts(tagged), **active}
            sound = {**tagged, **active}            # live + active-only; passive never reaches OSC
            count = len(active)
        self._settings.num_ghosts = count
        self._emit(tagged, ghosts, sound)

    def _emit(self, tagged: FrameDict, ghosts: FrameDict, sound: FrameDict) -> None:
        for cb in self._frame_callbacks:
            cb(tagged)
        for cb in self._ghost_callbacks:
            cb(ghosts)
        for cb in self._sound_callbacks:
            cb(sound)

    # -- passive layer & dwell/motion (call under _lock) ---------------------

    def _sweep_sample(self, frames: FrameDict) -> None:
        """Two things per person, per tick:
        - **Break free (continuous):** the moment he is > ``band_degrees`` from his last-swept spot,
          commit his **last valid (held) pose** at the spot he held it (if it had settled) and drop the
          passive — so the ghost is his intentional held pose, never the pose he was in while leaving.
        - **Follow (on a playhead crossing):** refresh his passive ghost's spot/pose, build dwell/motion,
          and track pose stability (how long he's held the same pose).
        """
        band = math.radians(self._settings.band_degrees)
        for tid, frame in frames.items():
            az = frame[Azimuth].value

            # Break free — checked every tick, not just at a crossing.
            p = self._passive.get(tid)
            if (p is not None and not math.isnan(az)
                    and abs(math.atan2(math.sin(az - p.spot), math.cos(az - p.spot))) > band):
                if self._settled(p) and p.stable_frame is not None:
                    self._commit(p.stable_frame, p.stable_spot)   # freeze the last held pose at its spot
                del self._passive[tid]               # re-anchor a fresh passive on the next crossing

            # Follow — only when the playhead sweeps across him.
            offset = frame[PlayheadOffset].value
            prev = self._prev_offset.get(tid, float("nan"))
            self._prev_offset[tid] = offset      # updated every tick to catch the crossing
            hit = (not math.isnan(prev) and prev * offset < 0.0
                   and abs(prev) < _HALF_PI and abs(offset) < _HALF_PI)
            if not hit or math.isnan(az):
                continue
            p = self._passive.get(tid)
            if p is None:
                p = _Passive(spot=az, frame=frame, sweeps=1, anchor_motion=float("nan"),
                             stable_sweeps=1, stable_frame=None, stable_spot=float("nan"), stable_age=0)
                self._passive[tid] = p
            else:
                sim = _pose_similarity(frame[Angles], p.frame[Angles], self._settings.stability_scale)
                p.stable_sweeps = p.stable_sweeps + 1 if sim >= self._settings.stability_threshold else 1
                p.sweeps += 1          # still on the spot → follow him, build dwell
                p.spot = az
                p.frame = frame
            # A pose held long enough is "valid" (remembered); otherwise the last valid pose ages.
            if p.stable_sweeps >= self._settings.stability_sweeps:
                p.stable_frame = p.frame
                p.stable_spot = p.spot
                p.stable_age = 0
            elif p.stable_frame is not None:
                p.stable_age += 1
            if p.sweeps >= self._settings.motion_sweeps and math.isnan(p.anchor_motion):
                p.anchor_motion = frame[MotionTime].value   # motion begins counting after the warm-up
        # Drop per-person state for vanished tracks (vanishing leaves nothing).
        present = set(frames)
        for store in (self._prev_offset, self._passive):
            for tid in [t for t in store if t not in present]:
                del store[tid]

    def _settled(self, p: _Passive) -> bool:
        """Has the passive ghost settled — worth committing when it breaks free? Requires dwell full
        (``sweeps ≥ dwell_sweeps``), motion full, and a **valid, fresh** held pose: ``stable_frame`` set
        and last held no more than ``stability_max_age`` sweeps ago (so a stale pose leaves nothing)."""
        if p.sweeps < self._settings.dwell_sweeps or math.isnan(p.anchor_motion):
            return False
        if p.stable_frame is None or p.stable_age > self._settings.stability_max_age:
            return False
        mt = p.frame[MotionTime].value
        return not math.isnan(mt) and (mt - p.anchor_motion) / self._settings.motion_scale >= 1.0

    def _metrics(self, tid: int, frame: Frame) -> tuple[float, float, float]:
        """Continuous dwell (whole sweeps + sub-sweep phase from ``PlayheadOffset``), motion, and
        stability (held sweeps / ``stability_sweeps``, 1.0 = valid) for a live person, from his passive
        ghost. All zero if he has no passive ghost yet."""
        p = self._passive.get(tid)
        if p is None:
            return 0.0, 0.0, 0.0
        offset = frame[PlayheadOffset].value
        phase = 0.0 if math.isnan(offset) else ((-offset) % _TWO_PI) / _TWO_PI
        dwell = _clamp01((p.sweeps - 1 + phase) / max(1, self._settings.dwell_sweeps - 1))
        mt = frame[MotionTime].value
        motion = (0.0 if (math.isnan(p.anchor_motion) or math.isnan(mt))
                  else _clamp01((mt - p.anchor_motion) / self._settings.motion_scale))
        stability = _clamp01(p.stable_sweeps / max(1, self._settings.stability_sweeps))
        return dwell, motion, stability

    def _tag_live(self, frames: FrameDict) -> FrameDict:
        """Stamp each live frame with its band-based ``GhostFeature`` (Dwell, Motion, Stability; Fade 1.0)."""
        out: FrameDict = {}
        for tid, frame in frames.items():
            dwell, motion, stability = self._metrics(tid, frame)
            out[tid] = replace(frame, {GhostFeature: GhostFeature.make(dwell, motion, stability, 1.0)})
        return out

    def _passive_ghosts(self, tagged: FrameDict) -> FrameDict:
        """One passive ghost per live person: his tagged (dwell/motion-carrying) frame marked
        ``PASSIVE``, keyed by his live id. Not sent to OSC; there for the light to visualise later."""
        state = GhostState.from_value(float(GhostStateValue.PASSIVE))
        return {tid: replace(tagged[tid], {GhostState: state})
                for tid in self._passive if tid in tagged}

    # -- active ghost machinery (call under _lock) ---------------------------

    def _advance_ghosts(self, playhead: float, frames: FrameDict) -> None:
        """Each time the playhead sweeps across an active ghost (its offset crosses zero): if a live
        pose overlaps it, **remove it** (a live person reclaims the spot); otherwise fade it by
        ``1/fade_sweeps`` and remove it once faded out. A just-made ghost has a NaN prev-offset, so it
        registers no crossing on its birth tick. Frozen while the playhead is NaN."""
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
        """Drop an active ghost and all its per-ghost state. Call under ``_lock``."""
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
        # Freeze the pose at the spot: pinned to the commit azimuth, marked ACTIVE, with a full (settled,
        # fully-present) GhostFeature — _live_ghosts refreshes its Fade each tick.
        self._ghosts[gid] = replace(reidentify(frame, gid),
                                    {Azimuth: Azimuth.from_value(az),
                                     GhostState: GhostState.from_value(float(GhostStateValue.ACTIVE)),
                                     GhostFeature: GhostFeature.make(1.0, 1.0, 1.0, 1.0)})
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
        """Active ghost id within ``band_degrees`` (radius) of ``az``, or ``None``. Call under ``_lock``."""
        band = math.radians(self._settings.band_degrees)
        for gid, gaz in self._ghost_az.items():
            d = az - gaz
            if abs(math.atan2(math.sin(d), math.cos(d))) <= band:
                return gid
        return None

    def _live_ghosts(self, playhead: float) -> FrameDict:
        """The frozen active-ghost registry, each stamped with its current ``GhostFeature`` (dwell/motion/
        stability pinned at 1, Fade decaying) and — when the playhead is finite — its ``PlayheadOffset``
        (from its fixed azimuth). ``GhostState.ACTIVE`` is carried on the stored frame and preserved. Call
        under ``_lock``."""
        out: FrameDict = {}
        for gid, g in self._ghosts.items():
            feats: dict = {GhostFeature: GhostFeature.make(1.0, 1.0, 1.0, self._ghost_fade[gid])}
            if not math.isnan(playhead):
                feats[PlayheadOffset] = PlayheadOffset.from_value(self._ghost_az[gid] - playhead)
            out[gid] = replace(g, feats)
        return out

    # -- settings ------------------------------------------------------------

    def _on_reset(self, _=None) -> None:
        self.clear()
        logger.info("Ghoster: cleared all ghosts")
