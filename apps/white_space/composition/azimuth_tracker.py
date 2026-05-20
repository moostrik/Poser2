"""AzimuthTracker — derives the rotating light's azimuth from fall signals.

Real mode (simulate=False):
    Each "fall" signal marks one full revolution completing.  The period is
    measured as the wall-clock time between consecutive falls.  Between falls
    azimuth advances as time_since_fall / measured_period, capped at 1.0 so
    it freezes rather than wrapping if the motor slows down or stops.

Simulate mode (simulate=True):
    Uses the rpm setting to advance azimuth synthetically and wrap at 1.0,
    producing synthetic fall events.  Useful for development without hardware.
"""

from time import monotonic

from modules.settings import BaseSettings, Field, Widget


class AzimuthTrackerSettings(BaseSettings):
    simulate:       Field[bool]  = Field(False,                               description="Advance azimuth from rpm without hardware")
    simulation_rpm: Field[float] = Field(0.0, min=0.0, max=2400.0, step=0.1,  description="Motor speed used in simulate mode (RPM)")
    latency_ms:     Field[float] = Field(0.0, min=0.0, max=200.0,  step=0.5,  description="Signal latency compensation (ms) — pre-advances phase on each fall")
    phase_offset:   Field[float] = Field(0.0, min=0.0, max=1.0,   step=0.001, description="Azimuth zero-point offset (0–1)", newline=True)
    azimuth:        Field[float] = Field(0.0, min=0.0, max=1.0,   step=0.001, access=Field.READ, widget=Widget.slider, description="Current light azimuth (0–1)")


class AzimuthTracker:
    """Tracks the rotating light's angular position (azimuth 0.0–1.0).

    Call notify_fall() each time the hardware fall signal arrives.
    Call tick(dt) once per compositor tick to advance and read azimuth.
    """

    def __init__(self, settings: AzimuthTrackerSettings) -> None:
        self._settings          = settings
        self._measured_period:  float | None = None
        self._time_since_fall:  float        = 0.0
        self._last_fall_time:   float | None = None

    def notify_fall(self) -> None:
        """Record a hardware fall signal.  Updates the measured period and resets phase."""
        now = monotonic()
        if self._last_fall_time is not None:
            self._measured_period = now - self._last_fall_time
        self._last_fall_time  = now
        self._time_since_fall = self._settings.latency_ms / 1000.0

    def tick(self, dt: float) -> float:
        """Advance internal state by dt seconds and return the current azimuth."""
        self._time_since_fall += dt
        offset = self._settings.phase_offset

        if self._settings.simulate:
            rpm = self._settings.simulation_rpm
            if rpm > 0.0:
                period = 60.0 / rpm
                # Wrap time accumulator so it never grows without bound
                if self._time_since_fall >= period:
                    self._time_since_fall %= period
                azimuth = (self._time_since_fall / period + offset) % 1.0
            else:
                azimuth = 0.0
        else:
            if self._measured_period is not None:
                raw     = self._time_since_fall / self._measured_period
                azimuth = (min(raw, 1.0) + offset) % 1.0
            else:
                azimuth = 0.0

        self._settings.azimuth = azimuth
        return azimuth
