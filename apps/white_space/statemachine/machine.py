"""StateMachine — the installation's single decision maker.

A condition-driven state machine that plays the states designed in ``data/STATES.md``
(the source of truth): a sequencer hybrid, progress-driven *within* a state and
condition-driven *between* states. Each tick (on the Conductor's light thread) it builds a
``StateContext`` from the board (participants, sync, hit-by-light, the playhead's content
clock and regime signals), lets the active state return its mix, evaluates that state's
transition conditions, and emits a ``SequencerState`` snapshot for the board and OSC sound.

The machine is the only component that talks to the Conductor, through three explicit
command callables: ``set_mix`` (the mix), ``reset_layers`` (explicit layer resets), and
``set_motor`` (the motor command; ``None`` relinquishes to the manual ``motor.mode`` setting).

``SequencerState`` is a boundary wire format only — reusing it keeps the board/OSC wiring
and the Max-facing contract identical to hd_trio's; its legacy ``stage`` field naming stays
at the wire, the internals speak ``state`` throughout.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import IntEnum, auto
from threading import Lock
from typing import Callable

from modules.session import SequencerState
from modules.settings import BaseSettings, Field, Widget
from modules.pose.analytics import SimilarityResult
from modules.utils import HotReloadMethods

from ..board import Board
from ..light import LightSettings, LayerId, Mix, MotorMode
from ..pose import PlayheadOffset

import logging
logger = logging.getLogger(__name__)


class SyncMode(IntEnum):
    """How many participants must be in sync for INTRO → INTRO_PLAY."""
    THREE         = 0   # at least 3 participants ≥ sync_threshold
    ALL_MINUS_ONE = auto()   # all but one
    ALL           = auto()   # everyone

    def required(self, participants: int) -> int:
        match self:
            case SyncMode.THREE:         return 3
            case SyncMode.ALL_MINUS_ONE: return max(participants - 1, 2)
            case _:                      return participants


class StateId(IntEnum):
    """The states of ``data/STATES.md`` (the source of truth). OFF = 0 is the
    operational off (outside the automatic graph, goto only; ``/global/state`` 0 means
    off on the wire). The *_INTRO / *_IDLE / *_PLAY entries are transitions promoted to
    states: their durations are the transition durations."""
    OFF        = 0
    IDLE       = auto()
    IDLE_INTRO = auto()
    INTRO      = auto()
    INTRO_IDLE = auto()
    INTRO_PLAY = auto()
    PLAY       = auto()
    END        = auto()
    END_INTRO  = auto()
    END_IDLE   = auto()


class StateMachineSettings(BaseSettings):
    """Configuration for the show StateMachine. Every timing is named for the state it
    times, unit in the name (seconds = wall clock, bars = playhead bars)."""
    enabled: Field[bool]      = Field(True,  description="Evaluate transitions automatically")
    hold:    Field[bool]      = Field(False, description="Freeze transitions; the active state keeps updating")
    session: Field[bool]      = Field(False, description="Session mode: timed overrides for the open-ended states (INTRO, PLAY)")
    select:  Field[StateId] = Field(StateId.IDLE, description="State to jump to with the goto button", newline=True)
    goto:    Field[bool]      = Field(False, widget=Widget.button, description="Jump to the selected state now (also when disabled)")

    # Transition-state durations — one per state. END_INTRO/END_IDLE have none: their
    # duration IS the physical spin-down (exit = the re-lock at LOW; fade = spin_down).
    intro_idle_bars:       Field[float] = Field(1.0,  min=0.1, max=20.0,  step=0.1, description="INTRO_IDLE: playhead bars back to IDLE", newline=True)
    intro_play_seconds:    Field[float] = Field(14.0, min=1.0, max=60.0,  step=0.5, description="INTRO_PLAY: spin-up transition (seconds)")
    end_bars:              Field[float] = Field(3.0,  min=0.5, max=20.0,  step=0.5, description="END: wind-down playhead bars (bidirectional ramp)")

    # Session-mode timeouts — named for the state they cut short
    intro_session_seconds: Field[float] = Field(60.0,  min=5.0, max=600.0,  step=1.0, description="Session: INTRO → INTRO_PLAY after this time", newline=True)
    play_session_seconds:  Field[float] = Field(150.0, min=5.0, max=1200.0, step=1.0, description="Session: PLAY → END after this time")

    # Condition tunables
    sync_threshold:     Field[float] = Field(0.75, min=0.0, max=1.0, step=0.01, widget=Widget.slider, description="A participant counts as in sync at this pose similarity", newline=True)
    sync_mode:          Field[SyncMode] = Field(SyncMode.THREE, description="INTRO → INTRO_PLAY: how many participants must be in sync (3 / all−1 / all)")
    count_hold_seconds: Field[float] = Field(1.0,  min=0.0, max=10.0, step=0.1, description="Participant-count debounce: a new count must persist this long before conditions see it")

    # Telemetry (read-only)
    current:      Field[StateId] = Field(StateId.IDLE, access=Field.READ, description="Current show state", newline=True)
    progress:     Field[float]     = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Active state progress")
    participants: Field[int]       = Field(0, access=Field.READ, description="Debounced participant count")
    sync:         Field[float]     = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="Mean pose similarity")
    in_sync:      Field[int]       = Field(0, access=Field.READ, description="Participants currently at or above sync_threshold")


@dataclass
class StateContext:
    """Per-tick inputs handed to the active state — wall clock, content clock, and conditions."""
    elapsed: float          # seconds since state entry (one-way wall-clock ramps)
    bars:    float          # playhead bars since state entry (one-way bar ramps)
    dt:      float          # this tick's wall-clock delta (bidirectional ramps integrate these)
    dbar:    float          # this tick's bar delta
    participants: int       # debounced live participant count (ghosts excluded)
    sync:    float          # mean pose similarity (0..1)
    sync_count: int         # participants whose similarity is ≥ sync_threshold
    hit:     bool           # a live participant was passed by the playhead this tick
    session: bool           # session mode active — states consult it in needs_state_change()
    prev: StateId | None    # the state we arrived from (None at boot) — lets a transition
                            # state ramp from where the show actually was (no dips)
    motor_locked: bool      # the playhead has re-synced to the measured rotation at LOW
                            # (stale-proof "at low speed" — S8/S9's exit anchor)
    ring_formed: bool       # commanded HIGH and the falls have gone silent — the bar has
                            # physically blurred into the ring (S5's un-lock anchor)
    spin_down: float        # gated normalized deceleration 0..1 (1 at re-lock) — the
                            # spin-down fade IS this signal


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
        # Failsafe: the show ALWAYS starts in IDLE (motor LOW), regardless of the persisted
        # `select` value — that field is only the goto target. A preset saved mid-show must
        # never boot the machine into a HIGH-motor state.
        self._current: StateId = StateId.IDLE
        self._active = None                     # set on the first update() tick

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
        config.bind(StateMachineSettings.goto, self._on_goto)
        config.bind(StateMachineSettings.enabled, self._on_enabled)

        # Hot reload of the state classes: re-instantiate, keep elapsed/bars, re-command motor.
        self._state_reloader = HotReloadMethods(StateBase, True, True)
        self._state_reloader.add_file_changed_callback(self._rebuild_states)

    # -- Lifecycle -----------------------------------------------------------

    def stop(self) -> None:
        """Teardown — unbind the settings callbacks (no thread of its own)."""
        self._config.unbind(StateMachineSettings.goto, self._on_goto)
        self._config.unbind(StateMachineSettings.enabled, self._on_enabled)

    # -- Settings callbacks --------------------------------------------------

    def _on_goto(self, value: bool) -> None:
        if value:
            self._goto_requested = True

    def _on_enabled(self, value: bool) -> None:
        """Relinquish the motor to the manual `motor.mode` setting while disabled;
        re-command the active state's mode when re-enabled."""
        if not value:
            self._set_motor(None)
        elif self._active is not None:
            self._set_motor(self._active.MOTOR)

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
        sync_count = sum(1 for v in values if v >= self._config.sync_threshold)
        return StateContext(
            elapsed=now - self._entered_time,
            bars=signals.bars - self._entered_bars,
            dt=dt,
            dbar=signals.bars - self._prev_bars,
            participants=participants,
            sync=sync,
            sync_count=sync_count,
            hit=hit,
            session=self._config.session,
            prev=self._prev_state,
            motor_locked=signals.synced,
            ring_formed=signals.ring_formed,
            spin_down=signals.spin_down,
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

        if self._active is None:
            # Startup failsafe: always enter IDLE (see __init__) — `select` is not consulted.
            self._goto_requested = False
            self._switch(StateId.IDLE, now, dt, signals.bars, participants, hit, signals)
        elif self._goto_requested:
            self._goto_requested = False
            self._switch(StateId(int(self._config.select)), now, dt, signals.bars, participants, hit, signals)

        ctx = self._build_context(now, dt, signals, participants, hit)
        entries = self._active.update(ctx)
        self._set_mix(entries)

        if self._config.enabled and not self._config.hold:
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
        self._config.sync = ctx.sync
        self._config.in_sync = ctx.sync_count
        self._notify_state(SequencerState(
            stage=int(self._current),                       # wire-format naming (see module doc)
            stage_progress=p,
            progress=(int(self._current) + p) / len(StateId),
            elapsed=ctx.elapsed,
            active=self._config.enabled,
        ))

    def _switch(self, target: StateId, now: float, dt: float, bars_now: float,
                participants: int, hit: bool, signals) -> None:
        """exit() old → reset timers → command motor → enter() new (its mix composes in the
        same tick's update() that follows, before the frame renders)."""
        if self._active is not None:
            self._active.exit()
            logger.info("show state %s → %s", self._current.name, target.name)
            self._prev_state = self._current
        self._current = target
        self._entered_time = now
        self._entered_bars = bars_now
        self._prev_bars = bars_now
        self._active = self._states[target]
        if self._config.enabled:
            self._set_motor(self._active.MOTOR)   # while disabled the operator owns the motor
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
            active=self._config.enabled,
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
        if self._config.enabled:
            self._set_motor(self._active.MOTOR)
        logger.info("show states reloaded (current: %s)", self._current.name)
