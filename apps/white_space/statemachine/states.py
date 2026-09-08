"""The states — one class per StateId, from ``data/STATES.md`` (the source of truth).

Each state owns its outgoing transitions (``needs_state_change``, in priority order:
when several conditions are true the same tick, the first wins) and returns its mix every
tick (``update``: a weighted layer list the machine forwards to the Compositor). Steady
states return constant weights; transition states blend by their own ``progress`` — their
duration *is* the transition duration. S8/S9 are the exception: their fade lives in the
``wind_down`` layer (constant mix; the layer owns the regime-crossing dynamics) and their
duration is the spin-down plus one playhead bar after the motor lock.

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
from .machine import StateId, StateMachineSettings, StateContext, SyncMode


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

    MOTOR: MotorMode = MotorMode.LOW    # commanded via the machine's set_motor on entry

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
    """S0 — the installation is off: the machine stands still and the strip is dark.
    An operational state, not a show beat — outside the automatic graph entirely:
    entered and left only via the operator's goto. On the wire, /global/state 0 = off."""
    MOTOR = MotorMode.STOPPED

    def update(self, ctx: StateContext) -> Mix:
        return []                       # dark strip


class IdleState(StateBase):
    """S1 — IDLE. The white searchlight (playhead) spins slowly through the empty space,
    supported by an atmospheric soundscape that evokes curiosity and plays on both blue
    lamps."""
    MOTOR = MotorMode.LOW

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.playhead_low, 1.0), (LayerId.sound_light, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.participants > 0:
            return StateId.IDLE_INTRO
        return None


class IdleIntroState(StateBase):
    """S2 — IDLE_INTRO. Someone has entered. The searchlight keeps sweeping at full
    brightness, but the sound is already stirring: the pose instrument starts a little
    *before* the actual hit — this anticipation is the reason the state exists. When the
    bright beam strikes the person the intro begins: the line snaps to dim and the
    soundscape stops."""
    MOTOR = MotorMode.LOW

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.playhead_low, 1.0), (LayerId.sound_light, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.hit:
            return StateId.INTRO
        if ctx.participants == 0:       # left before being hit → wind back via INTRO_IDLE
            return StateId.INTRO_IDLE
        return None


class IntroState(StateBase):
    """S3 — INTRO. The pose instrument is introduced. Neutral poses give a glass ping;
    arms raised gives a heavy bass; all other arm positions give unique sounds. The dim
    playhead flashes bright as it crosses each participant."""
    MOTOR = MotorMode.LOW
    DIM = 0.4                           # the DIM line level (INTRO_IDLE fades back up from it)

    def enter(self, ctx: StateContext) -> None:
        self._reset_layers([LayerId.playhead_flash])   # no stale flash decay from a previous cycle

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.playhead_low, self.DIM), (LayerId.playhead_flash, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.participants == 0:       # before the session timeout: an empty room never spins up
            return StateId.INTRO_IDLE
        # Enough participants in sync (sync_mode: 3 / all−1 / all) launches the spin-up.
        required = SyncMode(int(self._config.sync_mode)).required(ctx.participants)
        if ctx.sync_count >= required and ctx.participants >= 3:
            return StateId.INTRO_PLAY
        if ctx.session and ctx.elapsed >= self._config.intro_session_seconds:
            return StateId.INTRO_PLAY
        return None


class PlayState(StateBase):
    """S6 — PLAY. The participants play the instrument, creating music and light
    patterns. The space between participants holding the same pose fills with light."""
    MOTOR = MotorMode.HIGH

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.pose_instrument, 1.0), (LayerId.playhead_high, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.participants < 3:
            return StateId.END
        if ctx.session and ctx.elapsed >= self._config.play_session_seconds:
            return StateId.END
        return None


# -- Transition states (ramps) ----------------------------------------------------

class IntroIdleState(StateBase):
    """S4 — INTRO_IDLE. The participants have left mid-intro. Over one bar the dim line
    fades back to the bright searchlight and the soundscape fades back in.

    Both channels ramp from where the show actually was on entry: from INTRO the line
    starts DIM and the sound visuals at 0; on the IDLE_INTRO pass-through (someone left
    before being hit) both are already at 1.0 — no dip, no blink."""
    MOTOR = MotorMode.LOW

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._start_lamp: float = IntroState.DIM
        self._start_sound: float = 0.0

    def enter(self, ctx: StateContext) -> None:
        from_intro = ctx.prev == StateId.INTRO
        self._start_lamp = IntroState.DIM if from_intro else 1.0
        self._start_sound = 0.0 if from_intro else 1.0

    def update(self, ctx: StateContext) -> Mix:
        e = _ease(self.progress(ctx))
        return [(LayerId.playhead_low, _lerp(self._start_lamp, 1.0, e)),
                (LayerId.sound_light, _lerp(self._start_sound, 1.0, e))]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.bars >= self._config.intro_idle_bars:
            return StateId.IDLE
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.bars / self._config.intro_idle_bars)


class IntroPlayState(StateBase):
    """S5 — INTRO_PLAY. The participants have synced their poses: the machine spins up.
    The pose instrument takes over from the line during the spin-up, and the sound
    enhances the accelerating chaos.

    White is a **hard mix at the un-lock**: the dim line holds unchanged from INTRO while
    the strip is still physically lamps; the moment the motor passes the sensor ceiling
    and the ring forms (``ctx.ring_formed``), the instrument and the playhead line snap
    in. Blue eases in from the un-lock over the remaining spin-up, reaching 1.0 at the
    PLAY hand-off (per-channel mix weights)."""
    MOTOR = MotorMode.HIGH

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._unlock_elapsed: float | None = None

    def enter(self, ctx: StateContext) -> None:
        # A new show cycle's instrument starts clean (patterns and sync fill alike).
        # Deliberately NOT in PLAY's enter — PLAY is re-entered from END's wind-back
        # and must inherit the running instrument.
        self._reset_layers([LayerId.pose_instrument])
        self._unlock_elapsed = None

    def update(self, ctx: StateContext) -> Mix:
        if self._unlock_elapsed is None and ctx.ring_formed:
            self._unlock_elapsed = ctx.elapsed          # the ring physically formed — hard mix now
        if self._unlock_elapsed is None:
            return [(LayerId.playhead_low, IntroState.DIM)]   # still lamps: hold INTRO's dim line
        remaining = max(self._config.spin_up_seconds - self._unlock_elapsed, 1e-6)
        blue = _ease((ctx.elapsed - self._unlock_elapsed) / remaining)
        return [(LayerId.pose_instrument, (1.0, blue)), (LayerId.playhead_high, 1.0)]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if ctx.elapsed >= self._config.spin_up_seconds:
            return StateId.PLAY
        return None

    def progress(self, ctx: StateContext) -> float:
        return _clamp(ctx.elapsed / self._config.spin_up_seconds)


class EndState(StateBase):
    """S7 — END. Fewer than three participants remain: the machine begins its end. Over N
    bars the light crosses to full white and the sound reflects it. If participants
    return, the white winds back and PLAY resumes — the ramp runs both ways, never
    jumping (at p = 0 the mix equals PLAY's, so the hand-over is seamless). In session
    mode the wind-back is disabled so a session always concludes."""
    MOTOR = MotorMode.HIGH

    def __init__(self, config: StateMachineSettings, light: LightSettings,
                 reset_layers: Callable[[list[LayerId]], None]) -> None:
        super().__init__(config, light, reset_layers)
        self._p: float = 0.0

    def enter(self, ctx: StateContext) -> None:
        self._p = 0.0

    def update(self, ctx: StateContext) -> Mix:
        forward = ctx.participants < 3 or ctx.session
        self._p = self._ramp(self._p, ctx.dbar / self._config.end_bars, forward)
        # One ramp gives both CSV behaviors: the flood crosses the white to full while the
        # blue fades out with the instrument (the instrument is the only blue source).
        return [(LayerId.pose_instrument, 1.0 - self._p),
                (LayerId.playhead_high, 1.0 - self._p),
                (LayerId.flood, _ease(self._p))]

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if self._p >= 1.0:
            return StateId.END_INTRO if ctx.participants > 0 else StateId.END_IDLE
        if self._p <= 0.0 and not ctx.session and ctx.participants >= 3:
            return StateId.PLAY
        return None

    def progress(self, ctx: StateContext) -> float:
        return self._p


class WindDownStateBase(StateBase):
    """Shared S8/S9 engine: the dying wall. The mix is constant — the ``wind_down`` layer
    (reset on entry) owns the whole fade, timed over its ``spin_down_seconds`` and
    guaranteed extinguished within one round after the motor re-locks at LOW — and the
    landing look sits underneath, revealed as the wall dies. Exit: one full playhead bar
    after the lock (the state outlives the spin-down by that round). Progress is the
    layer's own fade readout, so OSC stage_progress rides the actual fade."""
    MOTOR = MotorMode.LOW
    TARGET: StateId

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._lock_bars: float | None = None    # ctx.bars at the motor lock

    def enter(self, ctx: StateContext) -> None:
        self._reset_layers([LayerId.wind_down])  # restart the fade at the full wall
        self._lock_bars = None

    def needs_state_change(self, ctx: StateContext) -> StateId | None:
        if self._lock_bars is None and ctx.motor_locked:
            self._lock_bars = ctx.bars
        if self._lock_bars is not None and ctx.bars - self._lock_bars >= 1.0:
            return self.TARGET
        return None

    def progress(self, ctx: StateContext) -> float:
        return self._light.high_layers.wind_down.progress


class EndIntroState(WindDownStateBase):
    """S8 — END_INTRO. Participants remain, so the machine returns to the intro: the wall
    of white fades away during the spin-down, revealing the dim playhead line underneath,
    while the distortion sound disappears; the fade finishes — the back light
    extinguishing completely — within one round after the motor lock (see
    ``WindDownStateBase``). The dim line is in the mix from the start, so there is no
    splice and no seam into INTRO."""
    TARGET = StateId.INTRO

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.wind_down, 1.0), (LayerId.playhead_low, IntroState.DIM)]


class EndIdleState(WindDownStateBase):
    """S9 — END_IDLE. The space is empty: the wall of white fades away during the
    spin-down, revealing the bright searchlight line — the front lamp stays at full the
    whole way — while the distortion disappears and the searchlight soundscape returns.
    The sound visuals fade in on the wall's own fade readout."""
    TARGET = StateId.IDLE

    def update(self, ctx: StateContext) -> Mix:
        return [(LayerId.wind_down, 1.0), (LayerId.playhead_low, 1.0),
                (LayerId.sound_light, self.progress(ctx))]


# -- Registry (total over StateId) ----------------------------------------------

STATES: dict[StateId, type[StateBase]] = {
    StateId.OFF:        OffState,
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
