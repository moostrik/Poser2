"""Playhead — the LOW-speed content clock, derived from the motor.

A numerically-controlled oscillator (NCO/PLL) whose behavior depends on the motor's
active mode (it is the *content sweep*, which never runs faster than LOW):
  - STOPPED → holds its last position (frozen).
  - IDLE / LOW → free-runs at the motor's rpm and softly phase-locks to its measured phase.
  - HIGH → free-runs at `low_rpm` and ignores the motor's fast phase, so the content sweep
    continues seamlessly from LOW (the motor's HIGH speed is for the pixel system).
Its phase is never snapped or reset, so it stays continuous across mode/speed switches,
stalls, and NaN gaps — only the rate changes.

The motor is offset-agnostic; the playhead owns its single content-alignment `offset`
(constant → does not break continuity). Pixel compositions own their own offsets.
"""

import math

from modules.settings import BaseSettings, Field, Widget

from .motor import MotorState, MotorMode


def _wrap_to_pi(x: float) -> float:
    return (x + math.pi) % math.tau - math.pi


class PlayheadSettings(BaseSettings):
    offset:    Field[float] = Field(0.0,  min=-math.pi, max=math.pi, step=0.01,
                                    description="Playhead zero-point alignment (radians)")
    tracking:  Field[float] = Field(0.1,  min=0.0, max=1.0, step=0.01,
                                    description="How tightly the playhead tracks the measured motor phase (0=free-run, 1=snap)")
    phase:     Field[float] = Field(0.0,  min=-math.pi, max=math.pi, step=0.001,
                                    access=Field.READ, widget=Widget.slider, description="Continuous playhead (−π…π)")


class Playhead:
    """Continuous phase accumulator locked to the motor; exposes `.phase` (offset applied)."""

    def __init__(self, settings: PlayheadSettings) -> None:
        self._settings = settings
        self._phase: float = 0.0   # offset-free continuous NCO state

    def tick(self, dt: float, motor: MotorState) -> None:
        """Advance the content clock from the motor's active mode (see module docstring):
        STOPPED holds, IDLE/LOW follow the motor (free-run + soft-lock), HIGH free-runs at
        `low_rpm` ignoring the fast motor phase."""
        if motor.mode == MotorMode.STOPPED:
            pass                                              # hold last position (frozen)
        elif motor.mode == MotorMode.HIGH:
            # Keep sweeping at the LOW content rate; the motor's fast phase is for the pixels.
            self._phase = _wrap_to_pi(self._phase + (motor.low_rpm / 60.0) * math.tau * dt)
        else:                                                 # IDLE, LOW — follow the motor
            rpm = motor.measured_rpm if motor.locked else motor.target_rpm
            self._phase += (rpm / 60.0) * math.tau * dt
            if motor.locked and not math.isnan(motor.phase):
                self._phase += self._settings.tracking * _wrap_to_pi(motor.phase - self._phase)
            self._phase = _wrap_to_pi(self._phase)
        self._settings.phase = self.phase

    @property
    def phase(self) -> float:
        """Continuous playhead in radians [-π, π), with the alignment offset applied."""
        return _wrap_to_pi(self._phase + self._settings.offset)

    def reset(self) -> None:
        self._phase = 0.0
