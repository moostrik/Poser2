"""MotorController — measures the rotating light's phase from fall signals and
echoes the commanded target speed. Offset-agnostic (raw measurement only); the
continuous content clock and any calibration offset live in the Playhead.

Real mode (simulate=False):
    Each "fall" signal marks one full revolution completing. The period is
    measured as the wall-clock time between consecutive falls; the phase is
    computed with sub-tick precision from monotonic(). If no fall arrives within
    one stall period the motor is declared unlocked (phase NaN).
    External callers use notify_fall(); while simulate=True it is a no-op.

Simulate mode (simulate=True):
    A stand-in for the physical motor + fall sensor, never a decision-maker: an internal
    thread ramps a simulated rpm toward whatever mode is *commanded* (same arbitration as
    the real motor) with modeled spin-up/down inertia, and fires synthetic falls — going
    silent above the sensor ceiling exactly like the real sensor, so measurement-driven
    behavior (lock, un-lock, deceleration) is identical in sim and reality.
"""

import math
from dataclasses import dataclass
from enum import IntEnum, auto
from threading import Thread, Event, Lock
from time import monotonic

from modules.settings import BaseSettings, Field, Widget

# The fall sensor is only valid below this rpm: no pulses are sent above it, and a reading above it
# (spinning down from HIGH) isn't trusted either — outside this range we trust the commanded speed.
_SENSOR_CEILING_RPM: float = 200.0

# A real revolution can't be faster than the sensor ceiling allows (60 / ceiling). A fall closer than
# this to the previous one is a repeated/duplicate signal (the one-sided sensor pulsed twice) — ignore
# it so it can't corrupt the measured period.
_MIN_FALL_INTERVAL_S: float = 60.0 / _SENSOR_CEILING_RPM

# Simulation loop cadence (s). The ramp + fall timing are sub-tick accurate regardless.
_SIM_TICK: float = 1.0 / 30.0 # same as the motor

# Simulated spin-up / spin-down rates (RPM/s). The real motor brakes ~3× faster than it
# accelerates: ≈6 s up to 2000, ≈2 s back down.
_SIM_ACCEL: float = 333.0
_SIM_DECEL: float = _SIM_ACCEL * 3.0


class MotorMode(IntEnum):
    """Commanded operating mode (the system sets it; target rpm is derived from it)."""
    STOPPED = auto()  # not spinning
    LOW     = auto()  # playhead sweep speed
    HIGH    = auto()  # fast spin (pixel content)


@dataclass
class MotorState:
    """Per-tick motor state: what it measures (phase, rpm) and what it commands (mode, rpm).
    Offset-agnostic — `phase` is the raw measured angle."""
    phase:         float     = 0.0                  # measured angular position, radians [-π, π)
    locked:        bool      = False                # a valid fall measurement exists this tick
    measured_rpm:  float     = 0.0                  # measured speed (0 when unlocked — sensor silent)
    effective_rpm: float     = 0.0                  # speed to act on: measured when locked, else the
                                                    # commanded target above the sensor ceiling, else 0
    mode:          MotorMode = MotorMode.STOPPED    # the driven (arbitrated) mode
    target_rpm:    float     = 0.0                  # commanded speed (sent to the motor by osc_light)
    low_rpm:       float     = 0.0                  # LOW-mode rpm — the playhead's content-sweep rate in HIGH
    raw_rpm:       float     = float('nan')         # last measured speed, regardless of trust (NaN = no falls yet)
    fall_age:      float     = float('inf')         # seconds since the last fall (inf = none) — silence above the
                                                    # ceiling is the un-lock evidence (the ring physically forming)


class MotorSettings(BaseSettings):
    simulate:             Field[bool] = Field(False,                              description="Simulate the motor + fall sensor (no hardware): obeys the commanded mode with modeled inertia")
    active_mode:          Field[MotorMode] = Field(MotorMode.STOPPED, access=Field.READ, description="Active mode — the arbitrated mode actually driven (debug > state machine; no command = STOPPED)")
    low_rpm:              Field[float] = Field(72.0,   min=0.0, max=300.0,  step=1.0,  description="Target rpm in LOW mode", newline=True)
    high_rpm:             Field[float] = Field(2000.0, min=0.0, max=2400.0, step=1.0,  description="Target rpm in HIGH mode")
    measured_rpm:         Field[float] = Field(0.0,   min=0.0, max=_SENSOR_CEILING_RPM, step=0.01,  access=Field.READ, description="Current measured RPM", newline=True)
    phase:                Field[float] = Field(0.0,   min=-math.pi, max=math.pi, step=0.001, access=Field.READ, widget=Widget.slider, description="Measured motor phase (−π…π)")


class MotorController:
    """Measures the rotating light's angular position (phase, radians [-π, π)) from
    falls and relays the commanded target speed.

    Call start() / stop() to manage the internal simulation thread.
    Call notify_fall() from external hardware signals (OSC/UDP).
    Call tick() once per render tick to read the measured state (the commanded rpm is
    derived from the `mode` setting).
    """

    def __init__(self, settings: MotorSettings) -> None:
        self._settings         = settings
        self._commanded:       MotorMode | None = None   # state machine command; None = relinquished (STOPPED)
        self._debug_mode:      MotorMode | None = None   # debug override (auto-follow); outranks the machine
        self._measured_period: float | None = None
        self._last_fall_time:  float | None = None
        self._fall_lock        = Lock()
        self._wakeup           = Event()
        self._running          = False
        self._sim_thread       = Thread(target=self._sim_loop, daemon=True, name="MotorSimulator")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._settings.bind(MotorSettings.simulate, self._on_sim_setting_changed)
        self._sim_thread.start()

    def stop(self) -> None:
        self._running = False
        self._settings.unbind(MotorSettings.simulate, self._on_sim_setting_changed)
        self._wakeup.set()
        self._sim_thread.join()

    # ------------------------------------------------------------------
    # Fall signal interface
    # ------------------------------------------------------------------

    def set_mode(self, mode: MotorMode | None) -> None:
        """State-machine command channel; ``None`` relinquishes — the motor stops (there
        is no manual mode: the machine and the debug select are the only authorities)."""
        self._commanded = mode

    def set_debug_mode(self, mode: MotorMode | None) -> None:
        """Debug override channel (the Conductor's auto-follow of the debug selection):
        outranks the state machine's command while set; ``None`` = debug off."""
        self._debug_mode = mode

    def notify_fall(self) -> None:
        """External hardware fall signal. No-op while simulating."""
        if self._settings.simulate:
            return
        self._fire_fall()

    def _fire_fall(self) -> None:
        """Record a real (hardware) fall at the current time."""
        self._fire_fall_at(monotonic())

    def _fire_fall_at(self, t: float) -> None:
        """Record a fall at time `t`, debouncing repeated/duplicate signals (and sub-tick sim falls)."""
        with self._fall_lock:
            last = self._last_fall_time
            if last is not None and t - last < _MIN_FALL_INTERVAL_S:
                # Too soon to be a real revolution (or out-of-order): a repeated/duplicate signal —
                # ignore it (keep last_fall_time at the real pulse) so it can't corrupt the period.
                return
            if last is not None:
                self._measured_period = t - last
            self._last_fall_time = t

    # ------------------------------------------------------------------
    # Simulation thread
    # ------------------------------------------------------------------

    def _on_sim_setting_changed(self, _value) -> None:
        """Wake the sim loop on a `simulate` change; clear fall history when switching it
        off so the first real hardware fall gets a clean measurement."""
        if not self._settings.simulate:
            with self._fall_lock:
                self._last_fall_time  = None
                self._measured_period = None
        self._wakeup.set()

    @staticmethod
    def _ramp_toward(current: float, target: float, max_step: float) -> float:
        """Move `current` toward `target` by at most `max_step` (models spin-up/down inertia)."""
        if current < target:
            return min(current + max_step, target)
        return max(current - max_step, target)

    def _sim_loop(self) -> None:
        """Internal thread: ramps a simulated rpm toward the sim mode's rpm and fires a
        synthetic fall each completed revolution (sub-tick timed), so spin-up/down is modelled.
        """
        current_rpm: float = 0.0
        revs:        float = 0.0   # revolutions accumulated toward the next fall
        last:        float = monotonic()

        while self._running:
            if not self._settings.simulate:
                current_rpm, revs = 0.0, 0.0          # real hardware drives the falls
                self._wakeup.wait(0.1)
                self._wakeup.clear()
                last = monotonic()
                continue

            now  = monotonic()
            dt   = now - last
            last = now

            target = self._target_rpm(self._target_mode())   # the commanded mode's rpm (0 = STOPPED → coast to stop)
            rate = _SIM_ACCEL if current_rpm < target else _SIM_DECEL   # brakes faster than it spins up
            current_rpm = self._ramp_toward(current_rpm, target, rate * dt)

            # Fire a synthetic fall per revolution — but only within the sensor's range: the real
            # sensor is silent above the ceiling, and the sim must be too, so measurement-driven
            # behavior (lock, un-lock, deceleration) is identical in sim and reality.
            if 0.0 < current_rpm <= _SENSOR_CEILING_RPM:
                revs += current_rpm / 60.0 * dt
                # Fire every completed revolution — not just one — so that when the fall rate
                # exceeds the loop rate (high rpm), revs stays in [0,1) and the latest fall
                # timestamp stays at `now` instead of drifting into the past (→ spurious stall).
                while revs >= 1.0:
                    revs -= 1.0
                    self._fire_fall_at(now - revs / (current_rpm / 60.0))   # interpolated crossing
            else:
                revs = 0.0

            self._wakeup.wait(_SIM_TICK)
            self._wakeup.clear()

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def _target_rpm(self, mode: MotorMode) -> float:
        """Commanded rpm derived from the active mode."""
        match mode:
            case MotorMode.LOW:  return self._settings.low_rpm
            case MotorMode.HIGH: return self._settings.high_rpm
            case _:              return 0.0   # STOPPED → 0

    def _target_mode(self) -> MotorMode:
        """The mode we are driving toward — arbitration, each rung explicitly owned:
        debug (auto-followed from the debug selection) > state machine command >
        STOPPED (relinquished / before the machine's first tick — no command means no
        spin). The sim is never a source — it obeys this same arbitration. Sets
        `target_rpm` (sent to the motor); the *actual* mode is only confirmed once the
        motor responds (see `tick`)."""
        if self._debug_mode is not None:
            return self._debug_mode
        if self._commanded is not None:
            return self._commanded
        return MotorMode.STOPPED

    def _measure(self, now: float) -> tuple[bool, float, float, float, float]:
        """Phase + rpm from the fall timestamps — returns
        (have_measurement, rpm, phase, raw_rpm, fall_age).

        Pure read of the fall state; writes nothing. No falls yet → (False, 0.0, NaN, NaN, inf).
        There is no stall/stop detection: a long silence is never treated as "stopped" (the fall
        signal is too sparse/late for that) — the last measurement simply persists until a new
        fall updates it. `raw_rpm`/`fall_age` carry the untrusted raw evidence for regime
        detection (silence above the ceiling = the ring forming)."""
        with self._fall_lock:
            last_fall_time  = self._last_fall_time
            measured_period = self._measured_period

        fall_age = (now - last_fall_time) if last_fall_time is not None else float('inf')
        if last_fall_time is None or measured_period is None or measured_period <= 0.0:
            return False, 0.0, float('nan'), float('nan'), fall_age

        rpm   = 60.0 / measured_period
        raw   = (now - last_fall_time) / measured_period   # revolutions since the last fall
        phase = (min(raw, 1.0) * math.tau + math.pi) % math.tau - math.pi
        return True, rpm, phase, rpm, fall_age

    def tick(self) -> MotorState:
        """Report the motor state from the commanded mode, refined by fall measurements when available.

        Trusts the command: `mode` is always the commanded mode and `target_rpm` its rpm — there is no
        stall/stop detection (the fall signal is too sparse/late for that). Falls give a measured phase
        + rpm only while running *in the sensor's range*: below the ceiling (above it no falls arrive)
        and not commanded STOPPED. When measuring, `effective_rpm` is the real measured speed and the
        playhead tracks `phase`; otherwise `effective_rpm` falls back to the commanded `target_rpm` and
        `phase` is NaN (the playhead free-runs).
        """
        target_mode = self._target_mode()
        target_rpm  = self._target_rpm(target_mode)

        have_measurement, measured, phase, raw_rpm, fall_age = self._measure(monotonic())
        # A measurement is usable only in the sensor's range — both the commanded and the measured speed
        # below the ceiling (no falls above it; a >ceiling reading isn't trusted) — and not STOPPED.
        fast   = target_rpm > _SENSOR_CEILING_RPM or measured > _SENSOR_CEILING_RPM
        locked = have_measurement and not fast and target_mode != MotorMode.STOPPED
        measured_rpm  = measured if locked else 0.0
        effective_rpm = measured_rpm if locked else target_rpm
        phase         = phase if locked else float('nan')

        # Publish the measurement to the read-back settings.
        self._settings.phase        = phase if locked else 0.0
        self._settings.measured_rpm = measured_rpm
        self._settings.active_mode  = target_mode
        return MotorState(phase=phase, locked=locked, measured_rpm=measured_rpm, effective_rpm=effective_rpm,
                          mode=target_mode, target_rpm=target_rpm, low_rpm=self._settings.low_rpm,
                          raw_rpm=raw_rpm, fall_age=fall_age)
