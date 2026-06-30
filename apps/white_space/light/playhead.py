"""Playhead — the LOW-speed content clock, derived from the motor.

A numerically-controlled oscillator (NCO/PLL) whose behavior depends on the motor's
active mode (it is the *content sweep*, which never runs faster than LOW):
  - STOPPED → holds its last position (frozen); no playhead (`.phase` is NaN).
  - IDLE / LOW → tracks the measured rotation *while locked*: the sweep rate is the motor speed
    *averaged* over a few revolutions (`speed_smoothing`, so per-revolution timing jitter does not
    wobble the sweep), and the `tracking` gain eases the *position* onto the measured phase. With no
    measurement (motor disconnected / not turning) there is nothing real to track, so it holds (NaN).
  - HIGH → free-runs at `low_rpm` and ignores the motor's fast phase, so the content sweep
    continues seamlessly from LOW (the motor's HIGH speed is for the pixel system). HIGH is
    unmeasurable by design, so it is always "live" (the content sweep is the playhead).
This is the **internal** phase; it is never snapped or reset, so it stays continuous across
mode/speed switches, stalls, and NaN gaps — only the rate changes.

**Spin-down re-acquire** (HIGH → LOW/IDLE): leaving HIGH does *not* re-lock onto the measured
phase right away — the motor is still spinning fast (unmeasurable above the sensor ceiling, then
measuring *above* `low_rpm` as it brakes). The sweep keeps free-running at `low_rpm` until the
motor has slowed back to content speed, then the normal `tracking` gain eases the internal phase
onto the measured phase. The sweep stays "live" throughout the spin-down (never NaN).

Re-lock is two-stage to defeat the motor's lack of stall detection: above the sensor ceiling the
motor keeps reporting its *stale* pre-HIGH measurement (`locked` with `measured_rpm ≈ low_rpm`), so
trusting the first content-speed reading would re-lock instantly while the light is still spinning
fast (the content would then race at the resumed sensor rpm). We therefore wait until we have seen a
*fresh* above-content measurement (the real spin-down in progress) and only re-lock once it has since
settled to `measured_rpm ≤ low_rpm × (1 + _RESYNC_RPM_TOL)`.

The motor is offset-agnostic; the playhead owns its single content-alignment `offset`
(constant → does not break continuity). Pixel compositions own their own offsets.
"""

import math

from modules.settings import BaseSettings, Field, Widget
from modules.utils import EMAFilter

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
                                         description="Phase-lock gain — how tightly the playhead's position locks to the measured motor phase (0=free-run, 1=snap)")
    speed_smoothing:Field[float] = Field(0.5,  min=0.0, max=1.0, step=0.01,
                                         description="How much to average the measured motor speed feeding the sweep rate (0=raw, 1=heavy)")
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
        self._seen_fast: bool = False                # saw a fresh above-content measurement since leaving HIGH
        self._live:     bool  = False                # is there a real playhead this tick (locked, HIGH, or re-syncing)
        self._rpm_ema = EMAFilter(freq=30.0)         # averages the per-revolution measured speed (feed-forward)
        self._time:    float = 0.0                   # accumulated time for the EMA's dt-correction
        self._tracking_prev: bool = False            # was the previous tick the locked-tracking branch (to seed the EMA)

    def tick(self, dt: float, motor: MotorState) -> None:
        """Advance the internal content clock from the motor's active mode, gating the re-lock onto
        the measured phase until a spun-down motor has returned to content speed."""
        self._time += dt
        # Leaving HIGH arms the re-sync: keep free-running until the motor slows back to content speed
        # rather than snapping onto its still-too-fast (or stale, see below) measured phase.
        if self._prev_mode == MotorMode.HIGH and motor.mode != MotorMode.HIGH:
            self._resyncing, self._seen_fast = True, False
        self._prev_mode = motor.mode

        if motor.mode == MotorMode.HIGH or motor.mode == MotorMode.STOPPED:
            self._resyncing = False                       # HIGH free-runs anyway; STOPPED holds
        elif self._resyncing and motor.locked:
            # Two-stage gate: first catch a *fresh* above-content reading (the real spin-down — the
            # motor otherwise keeps reporting its stale pre-HIGH ≈low_rpm measurement), then re-lock
            # only once that has settled back to content speed.
            if motor.measured_rpm > motor.low_rpm * (1.0 + _RESYNC_RPM_TOL):
                self._seen_fast = True
            elif self._seen_fast:
                self._resyncing = False                   # motor reached content speed → re-lock

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
            self._tracking_prev = False                       # hold last position (frozen)
        elif motor.mode == MotorMode.HIGH or self._resyncing:
            # Free-run at the LOW content rate: HIGH ignores the fast motor; the re-sync waits out the
            # spin-down without snapping to the still-too-fast measured phase.
            self._internal = _wrap_to_pi(self._internal + (motor.low_rpm / 60.0) * math.tau * dt)
            self._tracking_prev = False
        elif motor.locked and not math.isnan(motor.phase):    # IDLE, LOW — track the measured rotation
            # Feed-forward at the *smoothed* speed: per-revolution measurements jitter, so averaging the
            # rate keeps the sweep steady; the low `tracking` gain then eases the phase onto the light.
            if not self._tracking_prev:
                self._rpm_ema.reset(motor.measured_rpm)       # seed on (re)acquire → no ramp-in
            self._rpm_ema.setAlpha(max(0.03, 1.0 - self._settings.speed_smoothing))   # 0=raw … 1=heavy
            rpm = self._rpm_ema(motor.measured_rpm, self._time)
            self._internal += (rpm / 60.0) * math.tau * dt
            self._internal += self._settings.tracking * _wrap_to_pi(motor.phase - self._internal)
            self._internal = _wrap_to_pi(self._internal)
            self._tracking_prev = True
        else:
            self._tracking_prev = False
        # (IDLE/LOW with no measurement (disconnected) → hold; `.phase` reports NaN, not live)

    @property
    def phase(self) -> float:
        """Playhead in radians [-π, π) with the alignment offset applied — the value the outside
        world consumes. NaN when there is no live playhead — STOPPED, or IDLE/LOW with no measurement
        (motor disconnected / not turning) — per the board's HasPlayhead contract. The internal sweep
        stays continuous underneath, and `settings.playhead` keeps the last finite position for the UI."""
        if not self._live:
            return float('nan')
        return _wrap_to_pi(self._internal + self._settings.phase * math.tau)
