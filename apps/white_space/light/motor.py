"""MotorController — measures the rotating light's phase from fall signals and
echoes the commanded target speed. Offset-agnostic (raw measurement only); the
continuous content clock and any calibration offset live in the Playhead.

Real mode (simulate=False):
    Each "fall" signal marks one full revolution completing. The period is
    measured as the wall-clock time between consecutive falls; the phase is
    computed with sub-tick precision from monotonic(). If no fall arrives within
    one stall period the motor is declared unlocked (phase NaN).
    External callers use notify_fall(); while simulate=True it is a no-op.

Simulate mode (simulate != OFF):
    An internal thread fires _fire_fall() at the selected sim mode's rpm — the phase is
    measured from those synthetic falls exactly like real ones.
"""

import math
from dataclasses import dataclass
from enum import IntEnum, auto
from threading import Thread, Event, Lock
from time import monotonic

from modules.settings import BaseSettings, Field, Widget

# Stall = no fall for this many revolutions at the last measured speed → motor stopped.
_STALL_MISSED_REVS: float = 3.0

# Simulation loop cadence (s). The ramp + fall timing are sub-tick accurate regardless.
_SIM_TICK: float = 1.0 / 30.0 # same as the motor

# Simulated spin-up/down acceleration (RPM/s); ≈6 s to reach 2000.
_SIM_ACCEL: float = 333.0


class MotorMode(IntEnum):
    """Commanded operating mode (the system sets it; target rpm is derived from it)."""
    STOPPED = auto()  # not spinning
    IDLE    = auto()  # slow idle rotation
    LOW     = auto()  # playhead sweep speed
    HIGH    = auto()  # fast spin (pixel content)


class MotorSimMode(IntEnum):
    """Simulation selector: OFF uses real hardware falls; any mode fires synthetic falls
    at that mode's rpm (STOPPED = none), so no separate simulate speed is needed."""
    OFF     = auto()
    STOPPED = auto()
    IDLE    = auto()
    LOW     = auto()
    HIGH    = auto()


@dataclass
class MotorState:
    """Per-tick motor state: what it measures (phase, rpm) and what it commands (mode, rpm).
    Offset-agnostic — `phase` is the raw measured angle."""
    phase:        float     = 0.0                  # measured angular position, radians [-π, π)
    locked:       bool      = False                # a valid fall measurement exists this tick
    measured_rpm: float     = 0.0                  # measured speed
    mode:         MotorMode = MotorMode.STOPPED    # commanded mode
    target_rpm:   float     = 0.0                  # commanded speed (sent to the motor by osc_light)


class MotorSettings(BaseSettings):
    simulate:             Field[MotorSimMode] = Field(MotorSimMode.OFF,                          description="OFF = real hardware; any mode simulates the motor at that mode's rpm")
    mode:                 Field[MotorMode] = Field(MotorMode.LOW,                     description="Commanded mode — the target rpm is derived from it")
    active_mode:          Field[MotorMode] = Field(MotorMode.LOW, access=Field.READ, description="Mode actually in effect (the sim selector overrides the command)", newline=True)
    idle_rpm:             Field[float] = Field(7.0,    min=0.0, max=60.0,   step=0.5,  description="Target rpm in IDLE mode", newline=True)
    low_rpm:              Field[float] = Field(72.0,   min=0.0, max=300.0,  step=1.0,  description="Target rpm in LOW mode")
    high_rpm:             Field[float] = Field(2000.0, min=0.0, max=2400.0, step=1.0,  description="Target rpm in HIGH mode")
    measured_rpm:         Field[float] = Field(0.0,   min=0.0, max=2400.0, step=0.1,  access=Field.READ, description="Current measured RPM", newline=True)
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

    def notify_fall(self) -> None:
        """External hardware fall signal. No-op while simulating."""
        if self._settings.simulate != MotorSimMode.OFF:
            return
        self._fire_fall()

    def _fire_fall(self) -> None:
        """Record a real (hardware) fall at the current time."""
        self._fire_fall_at(monotonic())

    def _fire_fall_at(self, t: float) -> None:
        """Record a fall that occurred at time `t` (used for sub-tick-accurate sim falls)."""
        with self._fall_lock:
            if self._last_fall_time is not None:
                self._measured_period = t - self._last_fall_time
            self._last_fall_time = t

    # ------------------------------------------------------------------
    # Simulation thread
    # ------------------------------------------------------------------

    def _sim_rpm(self) -> float:
        """Synthetic-fall rpm — the active (sim) mode's rpm; 0 for OFF/STOPPED."""
        if self._settings.simulate == MotorSimMode.OFF:
            return 0.0
        return self._target_rpm(self._active_mode())

    def _on_sim_setting_changed(self, _value) -> None:
        """Wake the sim loop on a `simulate` change; clear fall history when switching to
        OFF so the first real hardware fall gets a clean measurement."""
        if self._settings.simulate == MotorSimMode.OFF:
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
            if self._settings.simulate == MotorSimMode.OFF:
                current_rpm, revs = 0.0, 0.0          # real hardware drives the falls
                self._wakeup.wait(0.1)
                self._wakeup.clear()
                last = monotonic()
                continue

            now  = monotonic()
            dt   = now - last
            last = now

            target = self._sim_rpm()                  # mode rpm (0 = STOPPED → coast to stop)
            current_rpm = self._ramp_toward(current_rpm, target, _SIM_ACCEL * dt)

            if current_rpm > 0.0:
                revs += current_rpm / 60.0 * dt
                if revs >= 1.0:
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
        return {
            MotorMode.IDLE: self._settings.idle_rpm,
            MotorMode.LOW:  self._settings.low_rpm,
            MotorMode.HIGH: self._settings.high_rpm,
        }.get(mode, 0.0)   # STOPPED → 0

    def _active_mode(self) -> MotorMode:
        """The mode in effect: the sim selector overrides the commanded mode while simulating."""
        sim = self._settings.simulate
        return MotorMode[sim.name] if sim != MotorSimMode.OFF else self._settings.mode

    def tick(self) -> MotorState:
        """Measure the raw motor phase + rpm from falls; command the rpm derived from the mode.

        Offset-agnostic — returns the raw measured phase (radians [-π,π)) and a `locked`
        flag. When unlocked (stalled / no falls) phase is NaN and the playhead free-runs.

        While simulating, the `simulate` selector overrides the commanded `mode` so the
        whole system behaves as that mode (rpm, content, MotorState.mode).
        """
        mode         = self._active_mode()
        target_rpm   = self._target_rpm(mode)
        now          = monotonic()

        with self._fall_lock:
            last_fall_time  = self._last_fall_time
            measured_period = self._measured_period
            # Stall = no fall for several revolutions at the last measured speed → the motor
            # has stopped/slowed; unlock so the playhead free-runs. Self-adapting (fast →
            # quick, idle → patient); checked under the lock so a concurrent fall isn't wiped.
            if last_fall_time is not None and measured_period is not None:
                if now - last_fall_time > _STALL_MISSED_REVS * measured_period:
                    self._last_fall_time  = None
                    self._measured_period = None
                    last_fall_time        = None
                    measured_period       = None

        if last_fall_time is not None and measured_period is not None:
            locked = True
            rpm    = 60.0 / measured_period
            raw    = (now - last_fall_time) / measured_period   # revolutions since the last fall
            phase  = (min(raw, 1.0) * math.tau + math.pi) % math.tau - math.pi
            self._settings.phase = phase
        else:
            locked = False
            rpm    = 0.0
            phase  = float('nan')
            self._settings.phase = 0.0

        self._settings.measured_rpm = rpm
        self._settings.active_mode  = mode
        return MotorState(phase=phase, locked=locked, measured_rpm=rpm, mode=mode, target_rpm=target_rpm)
