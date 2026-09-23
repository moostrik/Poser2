"""The states — one class per StateId, from ``docs/STATES.md`` (the source of truth).

Each state owns its outgoing transitions (``needs_state_change``, in priority order:
when several conditions are true the same tick, the first wins) and returns its mix every
tick (``update``: a weighted layer list the machine forwards to the Compositor). Steady
states return constant weights; transition states blend by their own ``progress`` — their
duration *is* the transition duration. S9/S10 are the exception: their fade lives in the
``beam_wind_down`` layer (constant mix; the layer owns the timed fade) and they exit once that
fade is complete and the playhead lock holds.

Mix-authoring rules:
- A layer at weight 0.0 stays *in* the returned list while it is still part of the look;
  omit it only when done (nothing resets implicitly — resets are explicit via
  ``reset_layers`` in ``enter()``).
- Parameters (layer settings, master): own every one you depend on — snap it in ``enter()``
  or ramp it from its captured current value in ``update()``; never silently assume it.

This module hot-reloads while the app runs (edit, save, the machine re-instantiates).
"""

from __future__ import annotations

from typing import Callable

import pytweening

from ..light import LightSettings, LayerId, Mix, MotorMode
from .settings import StateId, StateMachineSettings
from .machine import StateContext


# -- Easing helpers (weight curves over progress p in [0, 1]; pytweening easings) --

def _clamp(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def _ease(t: float) -> float:
    """Sine ease-in-out on [0,1]."""
    return pytweening.easeInOutSine(_clamp(t))


# -- Base -----------------------------------------------------------------------

class StateBase:
    """One show state: commands ``MOTOR`` on entry, returns its look every tick, and
    answers ``needs_state_change`` (the target state, or None to stay — callers must test
    ``is not None``: StateId.IDLE == 0 is falsy)."""

    MOTOR: MotorMode = MotorMode.BEAM    # commanded via the machine's set_motor on entry

    def __init__(self, config: StateMachineSettings, light: LightSettings,
                 reset_layers: Callable[[list[LayerId]], None]) -> None:
        self._config = config
        self._light = light
        self._reset_layers = reset_layers

    def enter(self, ctx: StateContext) -> None:
        """One-time setup on entry: optional reset_layers([...]) for a fresh start,
        capture parameter values to ramp from."""

    def update(self, ctx: StateContext) -> Mix:
        """Return this tick's look: [(layer, weight), ...]."""
        return []

    def exit(self) -> None:
        """One-time cleanup on leaving the state."""

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
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

class OffState(StateBase):
    """S0 — OFF. See docs/STATES.md."""
    MOTOR = MotorMode.BEAM

    def update(self, ctx: StateContext) -> Mix:
        return []                       # empty mix: dark

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.blackout or not ctx.is_playhead_locked:
            return None
        return StateId.OFF_IDLE


class IdleState(StateBase):
    """S2 — IDLE. See docs/STATES.md."""
    MOTOR = MotorMode.BEAM

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.beam_playhead, 1.0), (LayerId.beam_blue_sound, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.hit:                     # swept before the count debounce settled → the intro begins
            return StateId.INTRO
        if ctx.players > 0:
            return StateId.IDLE_INTRO
        return None


class IdleIntroState(StateBase):
    """S3 — IDLE_INTRO. See docs/STATES.md."""
    MOTOR = MotorMode.BEAM

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.beam_playhead, 1.0), (LayerId.beam_blue_sound, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.hit:
            return StateId.INTRO
        if ctx.players == 0:       # left before being hit → wind back via INTRO_IDLE
            return StateId.INTRO_IDLE
        return None


class IntroState(StateBase):
    """S4 — INTRO. See docs/STATES.md."""
    MOTOR = MotorMode.BEAM

    def enter(self, ctx: StateContext) -> None:
        self._reset_layers([LayerId.beam_flash])   # no stale flash decay from a previous cycle

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.beam_playhead, self._config.dim_level), (LayerId.beam_flash, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.players == 0:       # before the session timeout: an empty room never spins up
            return StateId.INTRO_IDLE
        # The players heard the same sound min_players times in a row: that many alike hits launch the spin-up.
        if ctx.sync_hits >= self._config.min_players and ctx.players >= self._config.min_players:
            return StateId.INTRO_PLAY
        if ctx.session and ctx.elapsed >= self._config.session.intro_seconds:
            return StateId.INTRO_PLAY
        return None


class PlayState(StateBase):
    """S7 — PLAY. See docs/STATES.md."""
    MOTOR = MotorMode.PROJECTION

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.pose_instrument, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.session:                 # a session plays out its time, whatever the count
            if ctx.elapsed >= self._config.session.play_seconds:
                return StateId.END
            return None
        if ctx.players < self._config.min_players:
            return StateId.END
        return None


# -- Transition states (ramps) ----------------------------------------------------

class OffIdleState(StateBase):
    """S1 — OFF_IDLE. See docs/STATES.md."""
    MOTOR = MotorMode.BEAM

    def update(self, ctx: StateContext) -> Mix:
        e = _ease(self.progress(ctx))
        return [(LayerId.beam_playhead, e), (LayerId.beam_blue_sound, e)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.hit:                     # swept mid-wake → the intro begins
            return StateId.INTRO
        if ctx.bars >= self._config.off_idle_bars:
            return StateId.IDLE
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.bars / self._config.off_idle_bars)


class IntroIdleState(StateBase):
    """S5 — INTRO_IDLE. See docs/STATES.md."""
    MOTOR = MotorMode.BEAM

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._start_lamp: float = self._config.dim_level
        self._start_sound: float = 0.0

    def enter(self, ctx: StateContext) -> None:
        # Ramp from where the show was: DIM line and no sound visuals from INTRO, both
        # already full on the IDLE_INTRO pass-through.
        from_intro = ctx.prev == StateId.INTRO
        self._start_lamp = self._config.dim_level if from_intro else 1.0
        self._start_sound = 0.0 if from_intro else 1.0

    def update(self, ctx: StateContext) -> Mix:
        e = _ease(self.progress(ctx))
        return [(LayerId.beam_playhead, _lerp(self._start_lamp, 1.0, e)),
                (LayerId.beam_blue_sound, _lerp(self._start_sound, 1.0, e))]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.hit:                     # someone returned and was swept mid-fade → back to the intro
            return StateId.INTRO
        if ctx.bars >= self._config.intro_idle_bars:
            return StateId.IDLE
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.bars / self._config.intro_idle_bars)


class IntroPlayState(StateBase):
    """S6 — INTRO_PLAY. See docs/STATES.md."""
    MOTOR = MotorMode.PROJECTION

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._projecting_elapsed: float | None = None

    def enter(self, ctx: StateContext) -> None:
        # A new show cycle's instrument starts clean (patterns and sync fill alike).
        # Deliberately NOT in PLAY's enter — PLAY is re-entered from END's wind-back
        # and must inherit the running instrument.
        self._reset_layers([LayerId.pose_instrument])
        self._projecting_elapsed = None

    def update(self, ctx: StateContext) -> Mix:
        if self._projecting_elapsed is None and ctx.is_projecting:
            self._projecting_elapsed = ctx.elapsed      # the projection shows — hard mix now
        if self._projecting_elapsed is None:
            return [(LayerId.beam_playhead, self._config.dim_level)]   # not projecting yet: hold INTRO's dim line
        remaining = max(self._config.spin_up_seconds - self._projecting_elapsed, 1e-6)
        blue = _ease((ctx.elapsed - self._projecting_elapsed) / remaining)
        return [(LayerId.pose_instrument, (1.0, blue))]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.elapsed >= self._config.spin_up_seconds:
            return StateId.PLAY
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.elapsed / self._config.spin_up_seconds)


class EndState(StateBase):
    """S8 — END. See docs/STATES.md."""
    MOTOR = MotorMode.PROJECTION

    def __init__(self, config: StateMachineSettings, light: LightSettings,
                 reset_layers: Callable[[list[LayerId]], None]) -> None:
        super().__init__(config, light, reset_layers)
        self._p: float = 0.0

    def enter(self, ctx: StateContext) -> None:
        self._p = 0.0

    def update(self, ctx: StateContext) -> Mix:
        forward = ctx.players < self._config.min_players or ctx.session
        self._p = self._ramp(self._p, ctx.dbar / self._config.end_bars, forward)
        # One ramp gives both CSV behaviors: the flood crosses the white to full while the
        # blue fades out with the instrument (the instrument is the only blue source).
        return [(LayerId.pose_instrument, 1.0 - self._p),
                (LayerId.flood, pytweening.easeOutSine(_clamp(self._p)))]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if self._p >= 1.0:
            return StateId.END_INTRO if ctx.players > 0 else StateId.END_IDLE
        if self._p <= 0.0 and not ctx.session and ctx.players >= self._config.min_players:
            return StateId.PLAY
        return None

    def progress(self, ctx: StateContext) -> float:
        return self._p


class WindDownStateBase(StateBase):
    """Shared S9/S10 engine: the dying wall. The mix is constant — the ``beam_wind_down`` layer
    (reset on entry) owns the whole fade, timed over its ``spin_down_seconds`` — and the
    landing look sits underneath, revealed as the wall dies. Exit: the fade complete and
    the playhead lock (the landing state needs a live playhead). Progress is the
    layer's own fade readout, so OSC stage_progress rides the actual fade."""
    MOTOR = MotorMode.BEAM
    TARGET: StateId

    def enter(self, ctx: StateContext) -> None:
        self._reset_layers([LayerId.beam_wind_down])  # restart the fade at the full wall

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.is_playhead_locked and self.progress(ctx) >= 1.0:
            return self.TARGET
        return None

    def progress(self, ctx: StateContext) -> float:
        return self._light.beam_layers.beam_wind_down.progress


class EndIntroState(WindDownStateBase):
    """S9 — END_INTRO. See docs/STATES.md."""
    TARGET = StateId.INTRO

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.beam_wind_down, 1.0), (LayerId.beam_playhead, self._config.dim_level)]


class EndIdleState(WindDownStateBase):
    """S10 — END_IDLE. See docs/STATES.md."""
    TARGET = StateId.IDLE

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.beam_wind_down, 1.0), (LayerId.beam_playhead, 1.0),
                (LayerId.beam_blue_sound, self.progress(ctx))]


# -- Registry (total over StateId) ----------------------------------------------

STATES: dict[StateId, type[StateBase]] = {
    StateId.OFF:        OffState,
    StateId.OFF_IDLE:   OffIdleState,
    StateId.IDLE:       IdleState,
    StateId.IDLE_INTRO: IdleIntroState,
    StateId.INTRO:      IntroState,
    StateId.INTRO_IDLE: IntroIdleState,
    StateId.INTRO_PLAY: IntroPlayState,
    StateId.PLAY:       PlayState,
    StateId.END:        EndState,
    StateId.END_INTRO:  EndIntroState,
    StateId.END_IDLE:   EndIdleState,
}
