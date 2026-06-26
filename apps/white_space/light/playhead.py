"""Playhead — the LOW-speed content clock, derived from the motor.

A numerically-controlled oscillator (NCO/PLL) whose behavior depends on the motor's
active mode (it is the *content sweep*, which never runs faster than LOW):
  - STOPPED → holds its last position (frozen).
  - IDLE / LOW → free-runs at the motor's rpm and softly phase-locks to its measured phase.
  - HIGH → free-runs at `low_rpm` and ignores the motor's fast phase, so the content sweep
    continues seamlessly from LOW (the motor's HIGH speed is for the pixel system).
This is the **internal** phase; it is never snapped or reset, so it stays continuous across
mode/speed switches, stalls, and NaN gaps — only the rate changes.

Optional **spin-up ride** (`high_follow`): on entering HIGH the *output* rides the physically
accelerating light (`= motor.phase`) until the motor passes `release_rpm`, then eases back to
the internal sweep over `release_smooth` seconds — a transition "whoosh". The internal phase
advances underneath the whole time, so the output reconverges onto the right content position.

The motor is offset-agnostic; the playhead owns its single content-alignment `offset`
(constant → does not break continuity). Pixel compositions own their own offsets.
"""

import math

from modules.settings import BaseSettings, Field, Widget

from .motor import MotorState, MotorMode


def _wrap_to_pi(x: float) -> float:
    return (x + math.pi) % math.tau - math.pi


def _lerp_angle(a: float, b: float, w: float) -> float:
    """Shortest-path angular blend: w=0 → a, w=1 → b."""
    return _wrap_to_pi(a + w * _wrap_to_pi(b - a))


class PlayheadSettings(BaseSettings):
    offset:         Field[float] = Field(0.0,  min=-math.pi, max=math.pi, step=0.01,
                                         description="Playhead zero-point alignment (radians)")
    tracking:       Field[float] = Field(0.1,  min=0.0, max=1.0, step=0.01,
                                         description="How tightly the playhead tracks the measured motor phase (0=free-run, 1=snap)")
    high_follow:    Field[bool]  = Field(False, newline=True,
                                         description="On entering HIGH, ride the accelerating motor through spin-up, then ease back to the low sweep")
    release_rpm:    Field[float] = Field(1200.0, min=0.0, max=2400.0, step=10.0,
                                         description="Ride the spin-up until the motor passes this RPM (≤ high_rpm)")
    release_smooth: Field[float] = Field(0.5,  min=0.0, max=5.0, step=0.05,
                                         description="Seconds to ease the ride back to the internal sweep")
    follow:         Field[float] = Field(0.0,  min=0.0, max=1.0, step=0.001,
                                         access=Field.READ, widget=Widget.slider, description="Ride weight (1=on the motor, 0=internal)")
    phase:          Field[float] = Field(0.0,  min=-math.pi, max=math.pi, step=0.001,
                                         access=Field.READ, widget=Widget.slider, description="Continuous playhead (−π…π)")


class Playhead:
    """Content clock derived from the motor; exposes `.phase` (offset applied). Holds an
    always-continuous internal sweep plus an optional spin-up ride toward the real motor."""

    def __init__(self, settings: PlayheadSettings) -> None:
        self._settings = settings
        self._internal: float = 0.0                  # offset-free continuous content clock
        self._output:   float = 0.0                  # what `.phase` exposes (internal, or the ride)
        self._prev_mode: MotorMode = MotorMode.STOPPED
        self._riding:   bool  = False                # spin-up ride active
        self._released: bool  = False                # past release_rpm → easing back
        self._w:        float = 0.0                  # ride weight (1=motor, 0=internal)

    def tick(self, dt: float, motor: MotorState) -> None:
        """Advance the internal content clock from the motor's active mode, then apply the
        optional spin-up ride to produce the output."""
        self._advance_internal(dt, motor)
        self._update_ride(dt, motor)
        self._settings.follow = self._w
        self._settings.phase  = self.phase

    def _advance_internal(self, dt: float, motor: MotorState) -> None:
        """The §10 mode-based content sweep (STOPPED holds, IDLE/LOW follow, HIGH free-runs low)."""
        if motor.mode == MotorMode.STOPPED:
            pass                                              # hold last position (frozen)
        elif motor.mode == MotorMode.HIGH:
            # Keep sweeping at the LOW content rate; the motor's fast phase is for the pixels.
            self._internal = _wrap_to_pi(self._internal + (motor.low_rpm / 60.0) * math.tau * dt)
        else:                                                 # IDLE, LOW — follow the motor
            rpm = motor.measured_rpm if motor.locked else motor.target_rpm
            self._internal += (rpm / 60.0) * math.tau * dt
            if motor.locked and not math.isnan(motor.phase):
                self._internal += self._settings.tracking * _wrap_to_pi(motor.phase - self._internal)
            self._internal = _wrap_to_pi(self._internal)

    def _update_ride(self, dt: float, motor: MotorState) -> None:
        """Spin-up ride: ride `motor.phase` from the entry-to-HIGH edge until `release_rpm`,
        then ease the weight to 0 over `release_smooth`. Output = blend(internal, motor)."""
        entered_high = motor.mode == MotorMode.HIGH and self._prev_mode != MotorMode.HIGH
        self._prev_mode = motor.mode

        if self._settings.high_follow and entered_high:
            self._riding, self._released, self._w = True, False, 1.0

        if self._riding and motor.mode != MotorMode.HIGH:
            self._riding, self._w = False, 0.0            # left HIGH → cancel

        if self._riding and not math.isnan(motor.phase):
            # Locked: ride the motor; advance the release once it passes the threshold.
            if not self._released and motor.measured_rpm >= self._settings.release_rpm:
                self._released = True
            if self._released:
                smooth = self._settings.release_smooth
                self._w = max(0.0, self._w - dt / smooth) if smooth > 0.0 else 0.0
                if self._w == 0.0:
                    self._riding = False
            self._output = _lerp_angle(self._internal, motor.phase, self._w)
        else:
            # Not riding, or riding but momentarily unlocked (e.g. the spin-up's first falls):
            # stay on the internal sweep and keep the ride armed until a lock appears.
            self._output = self._internal

    @property
    def phase(self) -> float:
        """Continuous playhead in radians [-π, π), with the alignment offset applied."""
        return _wrap_to_pi(self._output + self._settings.offset)

    def reset(self) -> None:
        self._internal = self._output = 0.0
        self._prev_mode = MotorMode.STOPPED
        self._riding = self._released = False
        self._w = 0.0
