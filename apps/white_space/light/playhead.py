"""Playhead — the BEAM-speed content clock, derived from the motor.

A numerically-controlled oscillator (NCO/PLL) whose behavior depends on the motor's
commanded mode (it is the *content sweep*, which never runs faster than BEAM):
  - STOPPED → holds its last position (frozen); no playhead (`.phase` is NaN).
  - IDLE / BEAM → tracks the measured rotation *while locked*: the sweep rate is the motor speed
    *averaged* over a few revolutions (`speed_smoothing`, so per-revolution timing jitter does not
    wobble the sweep), and the `tracking` gain eases the *position* onto the measured phase. With no
    measurement (motor disconnected / not turning) there is nothing real to track, so it holds (NaN).
  - PROJECTION → free-runs at `beam_rpm` and ignores the motor's fast phase, so the content sweep
    continues seamlessly from BEAM (the motor's PROJECTION speed is for the pixel system). PROJECTION is
    unmeasurable by design, so it is always "live" (the content sweep is the playhead).
This is the **internal** phase; it is never snapped or reset, so it stays continuous across
mode/speed switches, stalls, and NaN gaps — only the rate changes.

Each tick takes the motor's two halves separately: the `MotorMeasurement` (what the falls say)
and the `MotorCommand` in force over the dt being advanced (what the bar was told).

**Spin-down re-acquire** (PROJECTION → BEAM/IDLE): leaving PROJECTION does *not* re-lock onto the measured
phase right away — the motor is still spinning fast (unmeasurable above the sensor ceiling, then
measuring *above* `beam_rpm` as it brakes). The sweep keeps free-running at `beam_rpm` until the
motor has slowed back to content speed, then the normal `tracking` gain eases the internal phase
onto the measured phase. The sweep stays "live" throughout the spin-down (never NaN).

Re-lock is two-stage to defeat the motor's lack of stall detection: above the sensor ceiling the
motor keeps reporting its *stale* pre-PROJECTION measurement (`locked` with `measured_rpm ≈ beam_rpm`), so
trusting the first content-speed reading would re-lock instantly while the light is still spinning
fast (the content would then race at the resumed sensor rpm). We therefore wait until we have seen a
*fresh* above-content measurement (the real spin-down in progress) and only re-lock once it has since
settled to `measured_rpm ≤ beam_rpm × (1 + _RESYNC_RPM_TOL)`.

The motor is offset-agnostic; the playhead owns the single beam-mode calibration,
`pulse_offset` — the front lamp's azimuth at the sensor pulse, in degrees (constant → does
not break continuity). The projection offset that aligns the projection image lives in the light sender.
"""

import math

from modules.settings import BaseSettings, Field, Widget
from modules.utils import EMAFilter

from .motor import MotorMeasurement, MotorCommand, MotorMode, FIXTURE_PROJECTION_RPM

# Re-lock onto the measured phase once the spinning-down motor reaches content speed, within this
# relative tolerance of beam_rpm (absorbs measurement jitter as it settles at the BEAM target).
_RESYNC_RPM_TOL: float = 0.05

# Falls silent for this long while commanded PROJECTION → the bar spins fast enough for the projection
# image (the sensor cannot pulse above the ceiling; 2.5 ceiling-periods absorbs the last slow pulses).
_PROJECTING_SILENCE_S: float = 2.5 * 60.0 / FIXTURE_PROJECTION_RPM


def _wrap_to_pi(x: float) -> float:
    return (x + math.pi) % math.tau - math.pi


def _to_degrees(radians: float) -> float:
    """Radians (any range) → the operator's azimuth unit: degrees, 0–360."""
    return math.degrees(radians) % 360.0


class PlayheadSettings(BaseSettings):
    pulse_offset:   Field[float] = Field(0.0,  min=0.0, max=360.0, step=0.1,
                                         description="Playhead offset from the sensor pulse (degrees): the front lamp's azimuth at the pulse")
    tracking:       Field[float] = Field(0.1,  min=0.0, max=1.0, step=0.01,
                                         description="Phase-lock gain — how tightly the playhead's position locks to the measured motor phase (0=free-run, 1=snap)")
    speed_smoothing:Field[float] = Field(0.5,  min=0.0, max=1.0, step=0.01,
                                         description="How much to average the measured motor speed feeding the sweep rate (0=raw, 1=heavy)")
    playhead:       Field[float] = Field(0.0,  min=0.0, max=360.0, step=0.1,
                                         access=Field.READ, widget=Widget.slider, description="Continuous playhead azimuth (degrees)")


class Playhead:
    """Content clock derived from the motor; exposes `.phase` (offset applied). Holds an
    always-continuous internal sweep that free-runs at `beam_rpm` in PROJECTION and re-acquires the
    measured phase once the motor has spun back down to content speed."""

    def __init__(self, settings: PlayheadSettings) -> None:
        self._settings = settings
        self._internal: float = 0.0                  # offset-free continuous content clock
        self._bars:     float = 0.0                  # monotonic bar counter (1 bar = 1 full playhead cycle)
        self._prev_mode: MotorMode = MotorMode.STOPPED
        self._resyncing: bool = False                # left PROJECTION → free-running until the motor slows to content speed
        self._seen_fast: bool = False                # saw a fresh above-content measurement since leaving PROJECTION
        self._live:     bool  = False                # is there a real playhead this tick (locked, PROJECTION, or re-syncing)
        self._rpm_ema = EMAFilter(freq=30.0)         # averages the per-revolution measured speed (feed-forward)
        self._time:    float = 0.0                   # accumulated time for the EMA's dt-correction
        self._tracking_prev: bool = False            # was the previous tick the locked-tracking branch (to seed the EMA)
        self._is_projecting: bool = False            # mode signal: the bar spins fast enough for the projection image

    def tick(self, dt: float, motor: MotorMeasurement, command: MotorCommand) -> None:
        """Advance the internal content clock over ``dt`` from the command in force during it and
        the motor's measurement, gating the re-lock onto the measured phase until a spun-down
        motor has returned to content speed."""
        self._time += dt
        # Leaving PROJECTION arms the re-sync: keep free-running until the motor slows back to content speed
        # rather than snapping onto its still-too-fast (or stale, see below) measured phase.
        if self._prev_mode == MotorMode.PROJECTION and command.mode != MotorMode.PROJECTION:
            self._resyncing, self._seen_fast = True, False
        self._prev_mode = command.mode

        if command.mode == MotorMode.PROJECTION or command.mode == MotorMode.STOPPED:
            self._resyncing = False                       # PROJECTION free-runs anyway; STOPPED holds
        elif self._resyncing and motor.locked:
            # Two-stage gate: first catch a *fresh* above-content reading (the real spin-down — the
            # motor otherwise keeps reporting its stale pre-PROJECTION ≈beam_rpm measurement), then re-lock
            # only once that has settled back to content speed.
            if motor.measured_rpm > command.beam_rpm * (1.0 + _RESYNC_RPM_TOL):
                self._seen_fast = True
            elif self._seen_fast:
                self._resyncing = False                   # motor reached content speed → re-lock

        # A real playhead exists with a measurement (locked), in PROJECTION (free-run content sweep), or
        # while re-syncing (a live free-running sweep). IDLE/BEAM with no measurement → not live → NaN.
        self._live = command.mode == MotorMode.PROJECTION or motor.locked or self._resyncing
        self._advance_internal(dt, motor, command)
        self._update_regime_signals(motor, command)
        # Finite continuous position for the UI slider (`.phase` itself is NaN when not live).
        self._settings.playhead = _to_degrees(self._internal + math.radians(self._settings.pulse_offset))

    def _update_regime_signals(self, motor: MotorMeasurement, command: MotorCommand) -> None:
        """The physical mode-flip signals the show anchors on (the playhead owns them:
        it holds all the sync/resync/stale-reading knowledge).

        ``is_projecting`` (spin-up): commanded PROJECTION and the falls have gone silent — the sensor
        cannot pulse above the ceiling, so silence is the evidence the bar spins fast enough for
        the projection image. The spin-down side anchors on ``is_locked`` itself (the playhead
        lock): the sensor's spin-down readings don't resolve a usable deceleration ramp, so the
        S9/S10 fade is timed instead (the beam_wind_down layer)."""
        self._is_projecting = command.mode == MotorMode.PROJECTION and motor.fall_age > _PROJECTING_SILENCE_S

    def _advance_internal(self, dt: float, motor: MotorMeasurement, command: MotorCommand) -> None:
        """The mode-based content sweep (STOPPED holds, IDLE/BEAM track the measured phase, PROJECTION and
        the post-PROJECTION re-sync free-run at the BEAM content rate).

        The bar counter accumulates alongside: at the sweep's own rate while it advances, and at
        the commanded content rate while unmeasured (IDLE/BEAM with no falls) — so bar-denominated
        show durations never stall on a missing sensor. Only STOPPED holds the count."""
        if command.mode == MotorMode.STOPPED:
            self._tracking_prev = False                       # hold last position (frozen)
        elif command.mode == MotorMode.PROJECTION or self._resyncing:
            # Free-run at the BEAM content rate: PROJECTION ignores the fast motor; the re-sync waits out the
            # spin-down without snapping to the still-too-fast measured phase.
            self._internal = _wrap_to_pi(self._internal + (command.beam_rpm / 60.0) * math.tau * dt)
            self._bars += (command.beam_rpm / 60.0) * dt
            self._tracking_prev = False
        elif motor.locked and not math.isnan(motor.phase):    # BEAM — track the measured rotation
            # Feed-forward at the *smoothed* speed: per-revolution measurements jitter, so averaging the
            # rate keeps the sweep steady; the low `tracking` gain then eases the phase onto the light.
            if not self._tracking_prev:
                self._rpm_ema.reset(motor.measured_rpm)       # seed on (re)acquire → no ramp-in
            self._rpm_ema.setAlpha(max(0.03, 1.0 - self._settings.speed_smoothing))   # 0=raw … 1=heavy
            rpm = self._rpm_ema(motor.measured_rpm, self._time)
            self._internal += (rpm / 60.0) * math.tau * dt
            self._internal += self._settings.tracking * _wrap_to_pi(motor.phase - self._internal)
            self._internal = _wrap_to_pi(self._internal)
            self._bars += (rpm / 60.0) * dt
            self._tracking_prev = True
        else:
            # BEAM with no measurement (disconnected): the sweep holds (`.phase` NaN, not live),
            # but bars advance at the commanded content rate — the best estimate of the rotation.
            self._bars += (min(command.target_rpm, command.beam_rpm) / 60.0) * dt
            self._tracking_prev = False

    @property
    def phase(self) -> float:
        """Playhead in radians [-π, π) with the alignment offset applied — the value the outside
        world consumes. NaN when there is no live playhead — STOPPED, or IDLE/BEAM with no measurement
        (motor disconnected / not turning) — per the board's HasPlayhead contract. The internal sweep
        stays continuous underneath, and `settings.playhead` keeps the last finite position for the UI."""
        if not self._live:
            return float('nan')
        return _wrap_to_pi(self._internal + math.radians(self._settings.pulse_offset))

    @property
    def bars(self) -> float:
        """Monotonic bar counter (float, fractional): 1 bar = 1 full playhead cycle. The
        show's content clock — musical-timeline vocabulary, deliberately distinct from the
        machine's physical rotation. Never NaN; only STOPPED holds it."""
        return self._bars

    @property
    def is_locked(self) -> bool:
        """The playhead lock: true while the sweep is actively tracking the measured rotation at
        BEAM — stale-proof after a spin-down (re-lock gate passed)."""
        return self._tracking_prev

    @property
    def is_projecting(self) -> bool:
        """True while commanded PROJECTION with the falls gone silent — the bar spins fast enough
        for the projection image (S6's swap to the instrument)."""
        return self._is_projecting
