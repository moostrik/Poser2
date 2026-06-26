"""Playhead — the continuous content clock, derived from the motor.

A numerically-controlled oscillator (NCO/PLL): each tick it free-runs at the current
rpm and softly phase-locks to the motor's measured phase when a lock exists. Its phase
is **never reset**, so it stays continuous across mode/speed switches, stalls, and NaN
gaps — at a transition it holds and keeps winding, only the rate changes.

The motor is offset-agnostic; the playhead owns its single content-alignment `offset`
(constant → does not break continuity). Pixel compositions own their own offsets.
"""

import math

from modules.settings import BaseSettings, Field, Widget

from .motor import MotorState


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
        """Advance the NCO from the motor state: free-run at its rpm, then soft-lock to its
        measured phase when locked. Free-runs at the measured rpm when locked, else the
        commanded rpm (best estimate while the measurement is unavailable)."""
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
