"""Show states — one class per ShowState, straight from ``data/Stages-STAGES.csv``.

Each state owns its outgoing transitions (``needs_state_change``: the CSV's "goto Sx when…"
cells) and returns its look every tick (``update``: a weighted layer list the machine
forwards to the Compositor). Steady states return constant weights; transition states blend
by their own ``progress`` — their duration *is* the transition duration. Condition checks
are written in priority order: when several are true the same tick, the first wins.

Look-authoring rules:
- A layer at weight 0.0 stays *in* the returned list while it is still part of the look;
  omit it only when done (nothing resets implicitly — resets are explicit via
  ``reset_layers`` in ``enter()``).
- Parameters (layer settings, master): own every one you depend on — snap it in ``enter()``
  or ramp it from its captured current value in ``update()``; never silently assume it.

This module hot-reloads while the app runs (edit, save, the machine re-instantiates).
"""

from __future__ import annotations

import math
from typing import Callable

from ..light import LightSettings, LayerId, Look, MotorMode
from .machine import ShowState, StateMachineSettings, StateContext, SyncMode


# -- Easing helpers (weight curves over progress p in [0, 1]) -------------------

def _clamp(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def _ease(t: float) -> float:
    """Sine ease-in-out on [0,1]."""
    return 0.5 - 0.5 * math.cos(math.pi * _clamp(t))


# -- Base -----------------------------------------------------------------------

class StateBase:
    """One show state: commands ``MOTOR`` on entry, returns its look every tick, and
    answers ``needs_state_change`` (the target state, or None to stay — callers must test
    ``is not None``: ShowState.IDLE == 0 is falsy)."""

    MOTOR: MotorMode = MotorMode.LOW    # commanded via the machine's set_motor on entry

    def __init__(self, config: StateMachineSettings, light: LightSettings,
                 reset_layers: Callable[[list[LayerId]], None]) -> None:
        self._config = config
        self._light = light
        self._reset_layers = reset_layers

    def enter(self, ctx: StateContext) -> None:
        """One-time setup on entry: optional reset_layers([...]) for a fresh start,
        capture parameter values to ramp from."""

    def update(self, ctx: StateContext) -> Look:
        """Return this tick's look: [(layer, weight), ...]."""
        return []

    def exit(self) -> None:
        """One-time cleanup on leaving the state."""

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        """Do my exit conditions say we need a state change — and to which state?
        Never performs the switch; the machine does."""
        return None

    def progress(self, ctx: StateContext) -> float:
        """Progress through this state (0..1); 1.0 for open-ended steady states."""
        return 1.0

    @staticmethod
    def _ramp(p: float, delta: float, forward: bool) -> float:
        """Advance or wind back a bidirectional ramp by ``delta``, clamped to [0, 1]."""
        return _clamp(p + delta if forward else p - delta)


# -- Steady states ---------------------------------------------------------------

class IdleState(StateBase):
    """S0 — empty space, BRIGHT searchlight line sweeping."""
    MOTOR = MotorMode.LOW

    def update(self, ctx: StateContext) -> Look:
        return [(LayerId.playhead_lamp, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if ctx.participants > 0:
            return ShowState.IDLE_INTRO
        return None


class IdleIntroState(StateBase):
    """S1 — someone entered; BRIGHT line until the light hits them."""
    MOTOR = MotorMode.LOW

    def update(self, ctx: StateContext) -> Look:
        return [(LayerId.playhead_lamp, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if ctx.hit:
            return ShowState.INTRO
        if ctx.participants == 0:       # not in the CSV: left before being hit → wind back
            return ShowState.INTRO_IDLE
        return None


class IntroState(StateBase):
    """S2 — DIM line + BRIGHT flash when a participant is hit."""
    MOTOR = MotorMode.LOW
    DIM = 0.4                           # the DIM line level (INTRO_IDLE fades back up from it)

    def enter(self, ctx: StateContext) -> None:
        self._reset_layers([LayerId.playhead_flash])   # no stale flash decay from a previous cycle

    def update(self, ctx: StateContext) -> Look:
        return [(LayerId.playhead_lamp, self.DIM), (LayerId.playhead_flash, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if ctx.participants == 0:       # before the session timeout: an empty room never spins up
            return ShowState.INTRO_IDLE
        # Enough participants in sync (sync_mode: 3 / all−1 / all) launches the spin-up.
        required = SyncMode(int(self._config.sync_mode)).required(ctx.participants)
        if ctx.sync_count >= required and ctx.participants > 2:
            return ShowState.INTRO_PLAY
        if ctx.session and ctx.elapsed >= self._config.intro_session_seconds:
            return ShowState.INTRO_PLAY
        return None


class PlayState(StateBase):
    """S5 — full-speed pose instrument."""
    MOTOR = MotorMode.HIGH

    def update(self, ctx: StateContext) -> Look:
        return [(LayerId.pose_waves, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if ctx.participants < 3:
            return ShowState.END
        if ctx.session and ctx.elapsed >= self._config.play_session_seconds:
            return ShowState.END
        return None


# -- Transition states (ramps) ----------------------------------------------------

class IntroIdleState(StateBase):
    """S3 — fade the line back to BRIGHT over one playhead bar, back to IDLE.

    Ramps from where the show actually was: DIM arriving from INTRO (the CSV case),
    already-BRIGHT arriving from IDLE_INTRO (someone left before being hit — no dip)."""
    MOTOR = MotorMode.LOW

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._start: float = IntroState.DIM

    def enter(self, ctx: StateContext) -> None:
        self._start = IntroState.DIM if ctx.prev == ShowState.INTRO else 1.0

    def update(self, ctx: StateContext) -> Look:
        p = self.progress(ctx)
        return [(LayerId.playhead_lamp, _lerp(self._start, 1.0, _ease(p)))]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if ctx.bars >= self._config.intro_idle_bars:
            return ShowState.IDLE
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.bars / self._config.intro_idle_bars)


class IntroPlayState(StateBase):
    """S4 — spin-up: cross the line into the pose instrument over the spin-up time."""
    MOTOR = MotorMode.HIGH

    def enter(self, ctx: StateContext) -> None:
        # A new show cycle's instrument starts with clean wave history. Deliberately NOT in
        # PLAY's enter — PLAY is re-entered from END's wind-back and must inherit the waves.
        self._reset_layers([LayerId.pose_waves])

    def update(self, ctx: StateContext) -> Look:
        e = _ease(self.progress(ctx))
        return [(LayerId.playhead_lamp, 1.0 - e), (LayerId.pose_waves, e)]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if ctx.elapsed >= self._config.intro_play_seconds:
            return ShowState.PLAY
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.elapsed / self._config.intro_play_seconds)


class EndState(StateBase):
    """S6 — the bidirectional wind-down: while P < 3 the ramp advances toward the ending;
    while P ≥ 3 it winds back, and only at 0 does it hand over to PLAY (seamless — both
    looks are pose_waves at that point). In session mode the wind-back is disabled so the
    show always concludes."""
    MOTOR = MotorMode.HIGH

    def __init__(self, config: StateMachineSettings, light: LightSettings,
                 reset_layers: Callable[[list[LayerId]], None]) -> None:
        super().__init__(config, light, reset_layers)
        self._p: float = 0.0

    def enter(self, ctx: StateContext) -> None:
        self._p = 0.0

    def update(self, ctx: StateContext) -> Look:
        forward = ctx.participants < 3 or ctx.session
        self._p = self._ramp(self._p, ctx.dbar / self._config.end_bars, forward)
        return [(LayerId.pose_waves, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if self._p >= 1.0:
            return ShowState.END_INTRO if ctx.participants > 0 else ShowState.END_IDLE
        if self._p <= 0.0 and not ctx.session and ctx.participants >= 3:
            return ShowState.PLAY
        return None

    def progress(self, ctx: StateContext) -> float:
        return self._p


class EndIntroState(StateBase):
    """S7 — spin-down with people present: fade the instrument out and the DIM line in,
    then back to INTRO."""
    MOTOR = MotorMode.LOW

    def update(self, ctx: StateContext) -> Look:
        e = _ease(self.progress(ctx))
        return [(LayerId.pose_waves, 1.0 - e), (LayerId.playhead_lamp, IntroState.DIM * e)]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if ctx.elapsed >= self._config.end_intro_seconds:
            return ShowState.INTRO
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.elapsed / self._config.end_intro_seconds)


class EndIdleState(StateBase):
    """S8 — spin-down to an empty space: fade the instrument out and the BRIGHT line in,
    then back to IDLE."""
    MOTOR = MotorMode.LOW

    def update(self, ctx: StateContext) -> Look:
        e = _ease(self.progress(ctx))
        return [(LayerId.pose_waves, 1.0 - e), (LayerId.playhead_lamp, e)]

    def needs_state_change(self, ctx: StateContext) -> ShowState | None:
        if ctx.elapsed >= self._config.end_idle_seconds:
            return ShowState.IDLE
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.elapsed / self._config.end_idle_seconds)


# -- Registry (total over ShowState) ----------------------------------------------

STATES: dict[ShowState, type[StateBase]] = {
    ShowState.IDLE:       IdleState,
    ShowState.IDLE_INTRO: IdleIntroState,
    ShowState.INTRO:      IntroState,
    ShowState.INTRO_IDLE: IntroIdleState,
    ShowState.INTRO_PLAY: IntroPlayState,
    ShowState.PLAY:       PlayState,
    ShowState.END:        EndState,
    ShowState.END_INTRO:  EndIntroState,
    ShowState.END_IDLE:   EndIdleState,
}
