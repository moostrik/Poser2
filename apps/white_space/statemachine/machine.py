"""StateMachine — the installation's single decision maker.

A condition-driven state machine that plays the states designed in ``docs/STATES.md``
(the source of truth): a sequencer hybrid, progress-driven *within* a state and
condition-driven *between* states. Each tick (on the Conductor's light thread) it builds a
``StateContext`` from the board (the players present, the hit streak ``HitSync`` publishes — this
tick's hit and how many hits in a row struck alike poses — and the playhead's content clock and
lock signals), lets the active state return its mix, evaluates that state's transition conditions,
and emits a ``SequencerState`` snapshot for the board and OSC sound. It reads the board only; the
hits and their poses are ``pose/hit_sync.py``'s.

The machine is the only component that talks to the Conductor, through three explicit
command callables: ``set_mix`` (the mix), ``reset_layers`` (explicit layer resets), and
``set_motor`` (the motor command; ``None`` relinquishes — the motor stops).

``SequencerState`` is a boundary wire format only — reusing it keeps the board/OSC wiring
and the Max-facing contract identical to hd_trio's; its legacy ``stage`` field naming stays
at the wire, the internals speak ``state`` throughout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from modules.board import HitStreak
from modules.session import SequencerState
from modules.utils import HotReloadMethods

from ..board import Board
from ..light import LightSettings, LayerId, Mix, MotorMode
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
    players: int            # debounced live player count (ghosts excluded)
    sync_hits: int          # hits in a row, within one round, that struck alike poses (HitSync's streak)
    hit:     bool           # this tick the playhead is closest to a live player (the flash tick)
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

        self._state_callbacks: list[Callable[[SequencerState], None]] = []      # run in registration order

        # Tick timing
        self._prev_time: float | None = None
        self._entered_time: float = 0.0
        self._entered_bars: float = 0.0
        self._prev_bars: float = 0.0
        self._prev_state: StateId | None = None   # where the current state was entered from

        # Condition inputs
        self._eff_players: int | None = None   # debounced count (None until first tick)
        self._pending_count: int = 0
        self._pending_since: float = 0.0
        self._streak: HitStreak = HitStreak()       # this tick's hit and hit streak, from the board

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

    def _debounced_players(self, now: float, live_ids: set[int]) -> int:
        """Live player count — the people with a pose — debounced by count_hold_seconds so
        occlusion/re-acquisition flicker can't fire transitions."""
        raw = len(live_ids)
        if self._eff_players is None:
            self._eff_players = raw            # first tick: no startup delay
            self._pending_count = raw
        elif raw == self._eff_players:
            self._pending_count = raw               # settled — disarm any pending change
        elif raw != self._pending_count:
            self._pending_count = raw               # new candidate — start the hold window
            self._pending_since = now
        elif now - self._pending_since >= self._config.count_hold_seconds:
            self._eff_players = raw            # candidate held long enough
        return self._eff_players

    def _build_context(self, now: float, dt: float, signals,
                       players: int, hit: bool) -> StateContext:
        return StateContext(
            elapsed=now - self._entered_time,
            bars=signals.bars - self._entered_bars,
            dt=dt,
            dbar=signals.bars - self._prev_bars,
            players=players,
            sync_hits=self._streak.hits,
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
        players = self._debounced_players(now, set(frames.keys()))
        # This tick's hit and the hit streak: HitSync's, published on the board just before this tick.
        self._streak = self._board.get_hit_streak()
        hit = self._streak.hit

        if not self._entered:
            # Startup failsafe: always enter OFF (see __init__) — `manual.select` is not
            # consulted; OFF wakes through OFF_IDLE by itself once the playhead locks.
            self._goto_requested = False
            self._switch(StateId.OFF, now, dt, signals.bars, players, hit, signals)
        elif self._config.blackout and self._current != StateId.OFF:
            # Pinning blackout is OFF's entry door: highest-priority input, from anywhere,
            # beating hold and goto. Leaving OFF is a normal condition — OffState wakes
            # through OFF_IDLE once the toggle is released and the playhead lock holds.
            self._goto_requested = False
            self._switch(StateId.OFF, now, dt, signals.bars, players, hit, signals)
        elif self._goto_requested:
            self._goto_requested = False
            self._switch(StateId(int(self._config.manual.select)), now, dt, signals.bars, players, hit, signals)

        ctx = self._build_context(now, dt, signals, players, hit)
        entries = self._active.update(ctx)
        self._set_mix(entries)

        if not self._config.manual.hold:
            nxt = self._active.needs_state_change(ctx)
            if nxt is not None:
                self._switch(nxt, now, dt, signals.bars, players, hit, signals)
                ctx = self._build_context(now, dt, signals, players, hit)
                entries = self._active.update(ctx)
                self._set_mix(entries)

        self._prev_bars = signals.bars

        p = self._active.progress(ctx)
        self._config.progress = p
        self._config.players = players
        self._notify_state(SequencerState(
            stage=int(self._current),                       # wire-format naming (see module doc)
            stage_progress=p,
            progress=(int(self._current) + p) / len(StateId),
            elapsed=ctx.elapsed,
            active=True,           # wire-format legacy field; nothing consumes it
        ))

    def _switch(self, target: StateId, now: float, dt: float, bars_now: float,
                players: int, hit: bool, signals) -> None:
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
        self._active.enter(self._build_context(now, dt, signals, players, hit))

    # -- State callbacks (Sequencer-compatible) --------------------------------

    def add_state_callback(self, callback: Callable[[SequencerState], None]) -> None:
        if callback not in self._state_callbacks:
            self._state_callbacks.append(callback)
        callback(SequencerState(
            stage=int(self._current),
            stage_progress=self._config.progress,
            progress=(int(self._current) + self._config.progress) / len(StateId),
            elapsed=0.0,
            active=True,           # wire-format legacy field; nothing consumes it
        ))

    def remove_state_callback(self, callback: Callable[[SequencerState], None]) -> None:
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)

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
