"""StateMachine — the installation's single decision maker.

A condition-driven state machine that plays the states designed in ``docs/STATES.md``
(the source of truth): a sequencer hybrid, progress-driven *within* a state and
condition-driven *between* states. Each tick (on the Conductor's light thread) it builds a
``StateContext`` from the board (participants, hit-by-light, the pose frames' ``Similarity`` for the
sync condition, the playhead's content clock and lock signals), lets the active state return its
mix, evaluates that state's transition conditions, and emits a ``SequencerState`` snapshot for
the board and OSC sound.

The machine is the only component that talks to the Conductor, through three explicit
command callables: ``set_mix`` (the mix), ``reset_layers`` (explicit layer resets), and
``set_motor`` (the motor command; ``None`` relinquishes — the motor stops).

``SequencerState`` is a boundary wire format only — reusing it keeps the board/OSC wiring
and the Max-facing contract identical to hd_trio's; its legacy ``stage`` field naming stays
at the wire, the internals speak ``state`` throughout.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from modules.session import SequencerState
from modules.pose import features, FrameDict
from modules.utils import HotReloadMethods

from ..board import Board
from ..light import LightSettings, LayerId, Mix, MotorMode
from ..pose import PlayheadOffset, playhead_step, ticks_to_crossing
from .settings import StateId, StateMachineSettings, ManualSettings

import logging
logger = logging.getLogger(__name__)


_TINY = 1e-5   # the zero guard NormalizedScalarFeature uses for its harmonic mean


def _largest_sync_group(frames: FrameDict, threshold: float) -> tuple[int, float]:
    """The largest group of present participants in sync with each other — every member's harmonic mean of
    their ``Similarity`` toward the other members (strict: one poor match pulls it down) at or above
    ``threshold`` — as ``(size, the group's mean value)``; the best group of that size when several qualify,
    ``(0, 0.0)`` when no two are in sync. The rows already carry the neutral weight, so a person at neutral
    matches nobody and joins no group. Rows are indexed by player id; only the ids in ``frames`` are read, a
    member without data toward another member disqualifies the group."""
    ids = sorted(frames)
    rows = {i: frames[i][features.Similarity].values for i in ids}
    for size in range(len(ids), 1, -1):
        best: float | None = None
        for group in itertools.combinations(ids, size):
            values: list[float] = []
            for i in group:
                row = rows[i]
                others = [j for j in group if j != i]
                if any(j >= len(row) for j in others):
                    break
                s = row[others]
                if np.isnan(s).any():
                    break
                values.append(float(len(others) / np.sum(1.0 / np.maximum(s, _TINY))))
            if len(values) < size or min(values) < threshold:
                continue
            mean = float(np.mean(values))
            best = mean if best is None else max(best, mean)
        if best is not None:
            return size, best
    return 0, 0.0


@dataclass
class StateContext:
    """Per-tick inputs handed to the active state — wall clock, content clock, and conditions."""
    elapsed: float          # seconds since state entry (one-way wall-clock ramps)
    bars:    float          # playhead bars since state entry (one-way bar ramps)
    dt:      float          # this tick's wall-clock delta (bidirectional ramps integrate these)
    dbar:    float          # this tick's bar delta
    participants: int       # debounced live participant count (ghosts excluded)
    sync:    float          # mean similarity within the largest group in sync (0..1; 0 when no two are)
    sync_count: int         # size of the largest group in sync with each other (each member ≥ sync.threshold toward the others)
    hit:     bool           # this tick the playhead is closest to a live participant (the flash tick)
    session: bool           # session mode active — states consult it in needs_state_change()
    blackout: bool          # the pinned blackout toggle — OFF stays put while pinned and
                            # wakes through OFF_IDLE once released and the playhead lock holds
    prev: StateId | None    # the state we arrived from (None at boot) — lets a transition
                            # state ramp from where the show actually was (no dips)
    is_playhead_locked: bool  # playhead lock: the sweep tracks the measured rotation at BEAM
                              # (stale-proof after a spin-down — OFF and S9/S10 exit on it)
    is_projecting: bool     # PROJECTION commanded and the sensor silent: the bar is fast enough
                            # for the projection to show (S6's swap to the instrument)


class StateMachine:
    """Plays the show; see the module docstring. ``update()`` is ticked from
    ``conductor.add_update_callback`` — everything runs on the light thread and every input is
    read from the board there, so composing needs no locking."""

    def __init__(self, config: StateMachineSettings, light: LightSettings, board: Board,
                 set_mix: Callable[[Mix], None],
                 reset_layers: Callable[[list[LayerId]], None],
                 set_motor: Callable[[MotorMode | None], None],
                 pose_stage: int) -> None:
        self._config = config
        self._light = light
        self._board = board
        self._set_mix = set_mix
        self._reset_layers = reset_layers
        self._set_motor = set_motor
        self._pose_stage = pose_stage

        from .states import StateBase, STATES   # local import: states.py imports from this module
        self._states = {s: cls(config, light, reset_layers) for s, cls in STATES.items()}
        # Failsafe: the show ALWAYS starts in OFF (dark, motor BEAM) and wakes through OFF_IDLE
        # once the playhead has locked, regardless of the persisted `manual.select` value —
        # that field is only the goto target. A preset saved mid-show must never boot the
        # machine into a PROJECTION-motor state. `manual.hold` and `blackout` are forced off: a
        # preset saved mid-hold or mid-blackout must never freeze or strand the power-on
        # show — the installation always wakes into the show, never stays dark.
        config.manual.hold = False
        config.blackout = False
        self._current: StateId = StateId.OFF
        self._active: StateBase = self._states[self._current]
        self._entered: bool = False             # the boot entry into OFF happens on the first
                                                # update() tick, with real clock/bars timestamps

        self._state_callbacks: set[Callable[[SequencerState], None]] = set()

        # Tick timing
        self._prev_time: float | None = None
        self._entered_time: float = 0.0
        self._entered_bars: float = 0.0
        self._prev_bars: float = 0.0
        self._prev_state: StateId | None = None   # where the current state was entered from

        # Condition inputs
        self._eff_participants: int | None = None   # debounced count (None until first tick)
        self._pending_count: int = 0
        self._pending_since: float = 0.0
        self._prev_offsets: dict[int, float] = {}   # per-id PlayheadOffset for hit detection
        self._tick_sync: tuple[int, float] = (0, 0.0)   # this tick's largest group in sync: (size, mean similarity)

        self._goto_requested: bool = False
        config.manual.bind(ManualSettings.goto, self._on_goto)

        # Hot reload of the state classes: re-instantiate, keep elapsed/bars, re-command motor.
        self._state_reloader = HotReloadMethods(StateBase, True, True)
        self._state_reloader.add_file_changed_callback(self._rebuild_states)

    # -- Lifecycle -----------------------------------------------------------

    def stop(self) -> None:
        """Teardown — unbind the settings callbacks (no thread of its own)."""
        self._config.manual.unbind(ManualSettings.goto, self._on_goto)

    # -- Settings callbacks --------------------------------------------------

    def _on_goto(self, value: bool) -> None:
        if value:
            self._goto_requested = True

    # -- Inputs --------------------------------------------------------------

    def _sync_group(self, frames: FrameDict) -> tuple[int, float]:
        """The largest group in sync with each other, from the pose frames' ``Similarity``: (size, mean)."""
        return _largest_sync_group(frames, self._config.sync.threshold)

    def _debounced_participants(self, now: float, live_ids: set[int]) -> int:
        """Live participant count — the people with a pose — debounced by count_hold_seconds so
        occlusion/re-acquisition flicker can't fire transitions."""
        raw = len(live_ids)
        if self._eff_participants is None:
            self._eff_participants = raw            # first tick: no startup delay
            self._pending_count = raw
        elif raw == self._eff_participants:
            self._pending_count = raw               # settled — disarm any pending change
        elif raw != self._pending_count:
            self._pending_count = raw               # new candidate — start the hold window
            self._pending_since = now
        elif now - self._pending_since >= self._config.count_hold_seconds:
            self._eff_participants = raw            # candidate held long enough
        return self._eff_participants

    def _detect_hit(self, frames: FrameDict) -> bool:
        """True when this tick is the one the playhead is closest to a live participant — the tick a
        one-frame ``beam_flash`` lights (``ticks_to_crossing`` < ½ step at ``beam_rpm``). A
        PlayheadOffset sign flip + → − also counts, so a pass a jittered step skipped is still a hit;
        the ±π wrap flips − → +, so it never false-fires."""
        step = playhead_step(self._light.motor.beam_rpm, 1.0 / self._light.light_rate)
        hit = False
        offsets: dict[int, float] = {}
        for id, frame in frames.items():
            off = frame[PlayheadOffset].value
            if math.isnan(off):
                continue
            tau = ticks_to_crossing(off, step)
            prev = self._prev_offsets.get(id)
            if not math.isnan(tau) and abs(tau) < 0.5:
                hit = True
            elif prev is not None and prev > 0.0 and off < 0.0 and (prev - off) < math.pi:
                hit = True
            offsets[id] = off
        self._prev_offsets = offsets
        return hit

    def _build_context(self, now: float, dt: float, signals,
                       participants: int, hit: bool) -> StateContext:
        sync_count, sync = self._tick_sync
        return StateContext(
            elapsed=now - self._entered_time,
            bars=signals.bars - self._entered_bars,
            dt=dt,
            dbar=signals.bars - self._prev_bars,
            participants=participants,
            sync=sync,
            sync_count=sync_count,
            hit=hit,
            session=self._config.session.enabled,
            blackout=self._config.blackout,
            prev=self._prev_state,
            is_playhead_locked=signals.is_locked,
            is_projecting=signals.is_projecting,
        )

    # -- Tick (light thread, via conductor.add_update_callback) ---------------

    def update(self) -> None:
        now = time.time()
        dt = now - self._prev_time if self._prev_time is not None else 0.0
        self._prev_time = now
        signals = self._board.get_playhead_signals()

        # The people present are the people with a pose: `pose.tracklets.detection_timeout` decides.
        frames = self._board.get_frames(self._pose_stage)
        participants = self._debounced_participants(now, set(frames.keys()))
        hit = self._detect_hit(frames)
        self._tick_sync = self._sync_group(frames)

        if not self._entered:
            # Startup failsafe: always enter OFF (see __init__) — `manual.select` is not
            # consulted; OFF wakes through OFF_IDLE by itself once the playhead locks.
            self._goto_requested = False
            self._switch(StateId.OFF, now, dt, signals.bars, participants, hit, signals)
        elif self._config.blackout and self._current != StateId.OFF:
            # Pinning blackout is OFF's entry door: highest-priority input, from anywhere,
            # beating hold and goto. Leaving OFF is a normal condition — OffState wakes
            # through OFF_IDLE once the toggle is released and the playhead lock holds.
            self._goto_requested = False
            self._switch(StateId.OFF, now, dt, signals.bars, participants, hit, signals)
        elif self._goto_requested:
            self._goto_requested = False
            self._switch(StateId(int(self._config.manual.select)), now, dt, signals.bars, participants, hit, signals)

        ctx = self._build_context(now, dt, signals, participants, hit)
        entries = self._active.update(ctx)
        self._set_mix(entries)

        if not self._config.manual.hold:
            nxt = self._active.needs_state_change(ctx)
            if nxt is not None:
                self._switch(nxt, now, dt, signals.bars, participants, hit, signals)
                ctx = self._build_context(now, dt, signals, participants, hit)
                entries = self._active.update(ctx)
                self._set_mix(entries)

        self._prev_bars = signals.bars

        p = self._active.progress(ctx)
        self._config.progress = p
        self._config.participants = participants
        self._config.sync.similarity = ctx.sync
        self._config.sync.in_sync = ctx.sync_count
        self._notify_state(SequencerState(
            stage=int(self._current),                       # wire-format naming (see module doc)
            stage_progress=p,
            progress=(int(self._current) + p) / len(StateId),
            elapsed=ctx.elapsed,
            active=True,           # wire-format legacy field; nothing consumes it
        ))

    def _switch(self, target: StateId, now: float, dt: float, bars_now: float,
                participants: int, hit: bool, signals) -> None:
        """exit() old → reset timers → command motor → enter() new (its mix composes in the
        same tick's update() that follows, before the frame renders). The boot entry skips
        the exit half: nothing has been entered yet, and ``prev`` stays None."""
        if self._entered:
            self._active.exit()
            logger.info("show state %s → %s", self._current.name, target.name)
            self._prev_state = self._current
        self._entered = True
        self._current = target
        self._entered_time = now
        self._entered_bars = bars_now
        self._prev_bars = bars_now
        self._active = self._states[target]
        self._set_motor(self._active.MOTOR)
        self._config.current = target
        self._active.enter(self._build_context(now, dt, signals, participants, hit))

    # -- State callbacks (Sequencer-compatible) --------------------------------

    def add_state_callback(self, callback: Callable[[SequencerState], None]) -> None:
        self._state_callbacks.add(callback)
        callback(SequencerState(
            stage=int(self._current),
            stage_progress=self._config.progress,
            progress=(int(self._current) + self._config.progress) / len(StateId),
            elapsed=0.0,
            active=True,           # wire-format legacy field; nothing consumes it
        ))

    def remove_state_callback(self, callback: Callable[[SequencerState], None]) -> None:
        self._state_callbacks.discard(callback)

    def _notify_state(self, state: SequencerState) -> None:
        for cb in self._state_callbacks:
            try:
                cb(state)
            except Exception as e:
                logger.error(f"StateMachine state callback error: {e}")

    # -- Hot reload ------------------------------------------------------------

    def _rebuild_states(self) -> None:
        """Re-instantiate the state objects after a hot-reload of states.py: keep
        elapsed/bars, re-command the current state's motor mode (the look re-composes
        by itself on the next tick)."""
        from . import states as states_module
        self._states = {s: cls(self._config, self._light, self._reset_layers)
                        for s, cls in states_module.STATES.items()}
        self._active = self._states[self._current]
        self._set_motor(self._active.MOTOR)
        logger.info("show states reloaded (current: %s)", self._current.name)
