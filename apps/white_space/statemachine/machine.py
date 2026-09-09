"""StateMachine — the installation's single decision maker.

A condition-driven state machine that plays the states designed in ``data/STATES.md``
(the source of truth): a sequencer hybrid, progress-driven *within* a state and
condition-driven *between* states. Each tick (on the Conductor's light thread) it builds a
``StateContext`` from the board (participants, sync, hit-by-light, the playhead's content
clock and regime signals), lets the active state return its mix, evaluates that state's
transition conditions, and emits a ``SequencerState`` snapshot for the board and OSC sound.

The machine is the only component that talks to the Conductor, through three explicit
command callables: ``set_mix`` (the mix), ``reset_layers`` (explicit layer resets), and
``set_motor`` (the motor command; ``None`` relinquishes — the motor stops).

``SequencerState`` is a boundary wire format only — reusing it keeps the board/OSC wiring
and the Max-facing contract identical to hd_trio's; its legacy ``stage`` field naming stays
at the wire, the internals speak ``state`` throughout.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import Lock
from typing import Callable

from modules.session import SequencerState
from modules.pose.analytics import SimilarityResult
from modules.utils import HotReloadMethods

from ..board import Board
from ..light import LightSettings, LayerId, Mix, MotorMode
from ..pose import PlayheadOffset
from .settings import StateId, StateMachineSettings, ManualSettings

import logging
logger = logging.getLogger(__name__)


@dataclass
class StateContext:
    """Per-tick inputs handed to the active state — wall clock, content clock, and conditions."""
    elapsed: float          # seconds since state entry (one-way wall-clock ramps)
    bars:    float          # playhead bars since state entry (one-way bar ramps)
    dt:      float          # this tick's wall-clock delta (bidirectional ramps integrate these)
    dbar:    float          # this tick's bar delta
    participants: int       # debounced live participant count (ghosts excluded)
    sync:    float          # mean pose similarity (0..1)
    sync_count: int         # participants whose similarity is ≥ sync.threshold
    hit:     bool           # a live participant was passed by the playhead this tick
    session: bool           # session mode active — states consult it in needs_state_change()
    blackout: bool          # the pinned blackout toggle — OFF stays put while pinned and
                            # exits (to INTRO or IDLE by presence) once released
    prev: StateId | None    # the state we arrived from (None at boot) — lets a transition
                            # state ramp from where the show actually was (no dips)
    motor_locked: bool      # the playhead has re-synced to the measured rotation at LOW
                            # (stale-proof "at low speed" — S9/S10 exit one bar after it)
    ring_formed: bool       # commanded HIGH and the falls have gone silent — the bar has
                            # physically blurred into the ring (S6's un-lock anchor)


class StateMachine:
    """Plays the show; see the module docstring. ``update()`` is ticked from
    ``conductor.add_update_callback`` — everything runs on the light thread, so composing
    needs no locking (``set_similarity`` is the one cross-thread input and is locked)."""

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
        # Failsafe: the show ALWAYS starts in OFF (dark, motor LOW) and wakes through OFF_IDLE
        # once the playhead has locked, regardless of the persisted `manual.select` value —
        # that field is only the goto target. A preset saved mid-show must never boot the
        # machine into a HIGH-motor state. `manual.hold` and `blackout` are forced off: a
        # preset saved mid-hold or mid-blackout must never freeze or strand the power-on
        # show — the installation always wakes into the show, never stays dark.
        config.manual.hold = False
        config.blackout = False
        self._current: StateId = StateId.OFF
        self._active: StateBase = self._states[self._current]
        self._entered: bool = False             # the boot entry into IDLE happens on the first
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
        self._sync_lock = Lock()
        self._sync_values: list[float] = []

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

    def set_similarity(self, result: SimilarityResult) -> None:
        """Store the per-participant similarities; thread-safe (called from the analytics thread)."""
        values = [s.overall_similarity() for s in result.similarity.values()]
        values = [v for v in values if not math.isnan(v)]
        with self._sync_lock:
            self._sync_values = values

    def _debounced_participants(self, now: float) -> int:
        """Live participant count from the tracker, debounced by count_hold_seconds so
        occlusion/re-acquisition flicker can't fire transitions."""
        raw = sum(1 for t in self._board.get_tracklets().values() if t.is_active)
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

    def _detect_hit(self, live_ids: set[int]) -> bool:
        """True when a live participant's PlayheadOffset sign-flipped + → − this tick
        (the playhead swept past them). The ±π wrap flips − → +, so it never false-fires."""
        frames = self._board.get_frames(self._pose_stage)
        hit = False
        offsets: dict[int, float] = {}
        for id in live_ids:
            frame = frames.get(id)
            if frame is None:
                continue
            off = frame[PlayheadOffset].value
            if math.isnan(off):
                continue
            prev = self._prev_offsets.get(id)
            if prev is not None and prev > 0.0 and off < 0.0 and (prev - off) < math.pi:
                hit = True
            offsets[id] = off
        self._prev_offsets = offsets
        return hit

    def _build_context(self, now: float, dt: float, signals,
                       participants: int, hit: bool) -> StateContext:
        with self._sync_lock:
            values = self._sync_values
        sync = sum(values) / len(values) if values else 0.0
        sync_count = sum(1 for v in values if v >= self._config.sync.threshold)
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
            motor_locked=signals.synced,
            ring_formed=signals.ring_formed,
        )

    # -- Tick (light thread, via conductor.add_update_callback) ---------------

    def update(self) -> None:
        now = time.time()
        dt = now - self._prev_time if self._prev_time is not None else 0.0
        self._prev_time = now
        signals = self._board.get_playhead_signals()

        live_ids = {id for id, t in self._board.get_tracklets().items() if t.is_active}
        participants = self._debounced_participants(now)
        hit = self._detect_hit(live_ids)

        if not self._entered:
            # Startup failsafe: always enter OFF (see __init__) — `manual.select` is not
            # consulted; OFF wakes through OFF_IDLE by itself once the playhead locks.
            self._goto_requested = False
            self._switch(StateId.OFF, now, dt, signals.bars, participants, hit, signals)
        elif self._config.blackout and self._current != StateId.OFF:
            # Pinning blackout is OFF's entry door: highest-priority input, from anywhere,
            # beating hold and goto. Leaving OFF is a normal condition — OffState exits to
            # INTRO or IDLE (by presence) once the toggle is released.
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
