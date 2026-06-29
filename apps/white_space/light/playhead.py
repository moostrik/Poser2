"""Playhead — the LOW-speed content clock, derived from the motor.

A numerically-controlled oscillator (NCO/PLL) whose behavior depends on the motor's
active mode (it is the *content sweep*, which never runs faster than LOW):
  - STOPPED → holds its last position (frozen); no playhead (`.phase` is NaN).
  - IDLE / LOW → tracks the measured phase *while locked*. With no measurement (motor disconnected
    / not turning) there is nothing real to track, so the sweep holds and `.phase` is NaN.
  - HIGH → free-runs at `low_rpm` and ignores the motor's fast phase, so the content sweep
    continues seamlessly from LOW (the motor's HIGH speed is for the pixel system). HIGH is
    unmeasurable by design, so it is always "live" (the content sweep is the playhead).
This is the **internal** phase; it is never snapped or reset, so it stays continuous across
mode/speed switches, stalls, and NaN gaps — only the rate changes.

**Spin-down re-acquire** (HIGH → LOW/IDLE): leaving HIGH does *not* re-lock onto the measured
phase right away — the motor is still spinning fast (unmeasurable above the sensor ceiling, then
measuring *above* `low_rpm` as it brakes). The sweep keeps free-running at `low_rpm` until the
motor has slowed back to content speed (`measured_rpm ≤ low_rpm × (1 + _RESYNC_RPM_TOL)`), then
the normal `tracking` gain eases the internal phase onto the measured phase. The sweep stays
"live" throughout the spin-down (never NaN).

The motor is offset-agnostic; the playhead owns its single content-alignment `offset`
(constant → does not break continuity). Pixel compositions own their own offsets.
"""

import math

from modules.settings import BaseSettings, Field, Widget

from .motor import MotorState, MotorMode

# Re-lock onto the measured phase once the spinning-down motor reaches content speed, within this
# relative tolerance of low_rpm (absorbs measurement jitter as it settles at the LOW target).
_RESYNC_RPM_TOL: float = 0.05


def _wrap_to_pi(x: float) -> float:
    return (x + math.pi) % math.tau - math.pi


class PlayheadSettings(BaseSettings):
    phase:          Field[float] = Field(0.0,  min=0.0, max=1.0, step=0.01,
                                         description="Playhead zero-point alignment (0–1 turn)")
    tracking:       Field[float] = Field(0.1,  min=0.0, max=1.0, step=0.01,
                                         description="How tightly the playhead tracks the measured motor phase (0=free-run, 1=snap)")
    playhead:       Field[float] = Field(0.0,  min=-math.pi, max=math.pi, step=0.001,
                                         access=Field.READ, widget=Widget.slider, description="Continuous playhead (−π…π)")


class Playhead:
    """Content clock derived from the motor; exposes `.phase` (offset applied). Holds an
    always-continuous internal sweep that free-runs at `low_rpm` in HIGH and re-acquires the
    measured phase once the motor has spun back down to content speed."""

    def __init__(self, settings: PlayheadSettings) -> None:
        self._settings = settings
        self._internal: float = 0.0                  # offset-free continuous content clock
        self._prev_mode: MotorMode = MotorMode.STOPPED
        self._resyncing: bool = False                # left HIGH → free-running until the motor slows to content speed
        self._live:     bool  = False                # is there a real playhead this tick (locked, HIGH, or re-syncing)

    def tick(self, dt: float, motor: MotorState) -> None:
        """Advance the internal content clock from the motor's active mode, gating the re-lock onto
        the measured phase until a spun-down motor has returned to content speed."""
        # Leaving HIGH arms the re-sync: keep free-running until the motor slows back to content speed
        # rather than snapping onto its still-too-fast (or unmeasurable) phase.
        if self._prev_mode == MotorMode.HIGH and motor.mode != MotorMode.HIGH:
            self._resyncing = True
        self._prev_mode = motor.mode

        if motor.mode == MotorMode.HIGH or motor.mode == MotorMode.STOPPED:
            self._resyncing = False                       # HIGH free-runs anyway; STOPPED holds
        elif self._resyncing and motor.locked and motor.measured_rpm <= motor.low_rpm * (1.0 + _RESYNC_RPM_TOL):
            self._resyncing = False                       # motor reached content speed → re-lock

        # A real playhead exists with a measurement (locked), in HIGH (free-run content sweep), or
        # while re-syncing (a live free-running sweep). IDLE/LOW with no measurement → not live → NaN.
        self._live = motor.mode == MotorMode.HIGH or motor.locked or self._resyncing
        self._advance_internal(dt, motor)
        # Finite continuous position for the UI slider (`.phase` itself is NaN when not live).
        self._settings.playhead = _wrap_to_pi(self._internal + self._settings.phase * math.tau)

    def _advance_internal(self, dt: float, motor: MotorState) -> None:
        """The mode-based content sweep (STOPPED holds, IDLE/LOW track the measured phase, HIGH and
        the post-HIGH re-sync free-run at the LOW content rate)."""
        if motor.mode == MotorMode.STOPPED:
            pass                                              # hold last position (frozen)
        elif motor.mode == MotorMode.HIGH or self._resyncing:
            # Free-run at the LOW content rate: HIGH ignores the fast motor; the re-sync waits out the
            # spin-down without snapping to the still-too-fast measured phase.
            self._internal = _wrap_to_pi(self._internal + (motor.low_rpm / 60.0) * math.tau * dt)
        elif motor.locked and not math.isnan(motor.phase):    # IDLE, LOW — track the measured rotation
            self._internal += (motor.measured_rpm / 60.0) * math.tau * dt
            self._internal += self._settings.tracking * _wrap_to_pi(motor.phase - self._internal)
            self._internal = _wrap_to_pi(self._internal)
        # else IDLE/LOW with no measurement (disconnected) → hold; `.phase` reports NaN (not live)

    @property
    def phase(self) -> float:
        """Playhead in radians [-π, π) with the alignment offset applied — the value the outside
        world consumes. NaN when there is no live playhead — STOPPED, or IDLE/LOW with no measurement
        (motor disconnected / not turning) — per the board's HasPlayhead contract. The internal sweep
        stays continuous underneath, and `settings.playhead` keeps the last finite position for the UI."""
        if not self._live:
            return float('nan')
        return _wrap_to_pi(self._internal + self._settings.phase * math.tau)
