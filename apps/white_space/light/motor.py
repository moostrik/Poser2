"""MotorController — derives the rotating light's playhead from fall signals
and echoes the commanded target speed.

Real mode (simulate=False):
    Each "fall" signal marks one full revolution completing.  The period is
    measured as the wall-clock time between consecutive falls.  Playhead is
    computed with sub-tick precision directly from monotonic().  If no fall
    arrives within one low-speed revolution period, the motor is declared
    STOPPED and playhead resets to 0.
    External callers use notify_fall(); while simulate=True it is a no-op.

Simulate mode (simulate=True):
    An internal thread fires _fire_fall() at simulate_rpm cadence, bypassing
    the simulate gate.  All real-mode corrections (latency_ms, phase_offset,
    stall detection) apply identically.
"""

import math
from dataclasses import dataclass
from enum import IntEnum, auto
from threading import Thread, Event, Lock
from time import monotonic

from modules.settings import BaseSettings, Field, Widget


class MotorMode(IntEnum):
    STOPPED    = auto()  # motor not spinning; no fall signals
    LOW_SPEED  = auto()  # playhead tracking is meaningful
    HIGH_SPEED = auto()  # spinning too fast; playhead irrelevant, content switches


@dataclass
class MotorState:
    """Per-tick motor snapshot: rotation position, speed regime, commanded speed."""
    playhead:   float     = 0.0
    mode:       MotorMode = MotorMode.STOPPED
    target_rpm: float     = 0.0


class MotorSettings(BaseSettings):
    simulate:             Field[bool]  = Field(False,                                description="Advance playhead using internal simulation thread")
    simulate_rpm:         Field[float] = Field(0.0,   min=0.0, max=120.0,  step=0.1,  description="Motor speed used in simulate mode (RPM)")
    latency_ms:           Field[float] = Field(10.0,  min=0.0, max=100.0,  step=0.5,  description="Signal latency compensation (ms) — pre-advances phase on each fall", newline=True)
    phase_offset:         Field[float] = Field(0.0,   min=-math.pi, max=math.pi, step=0.01, description="Playhead zero-point offset (radians)")
    low_speed_threshold:  Field[float] = Field(1.0,   min=1,   max=100.0,  step=0.1,  description="RPM below which motor is declared STOPPED (stall detection)", newline=True)
    high_speed_threshold: Field[float] = Field(240.0, min=120, max=360.0,  step=1.0,  description="RPM above which motor mode switches to HIGH_SPEED")
    measured_rpm:         Field[float]    = Field(0.0,              min=0.0, max=2400.0, step=0.1,   access=Field.READ, description="Current measured RPM", newline=True)
    motor_mode:           Field[MotorMode] = Field(MotorMode.STOPPED,                               access=Field.READ, description="Current motor speed regime")
    playhead:             Field[float]    = Field(0.0,              min=-math.pi, max=math.pi, step=0.001, access=Field.READ, widget=Widget.slider, description="Current light playhead (−π…π)")


class MotorController:
    """Tracks the rotating light's angular position (playhead, radians [-π, π)) and
    relays the commanded target speed.

    Call start() / stop() to manage the internal simulation thread.
    Call notify_fall() from external hardware signals (OSC/UDP).
    Call tick(target_rpm) once per render tick to advance and read the state.
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
        self._settings.bind(MotorSettings.simulate,     self._on_sim_setting_changed)
        self._settings.bind(MotorSettings.simulate_rpm, self._on_sim_setting_changed)
        self._sim_thread.start()

    def stop(self) -> None:
        self._running = False
        self._settings.unbind(MotorSettings.simulate,     self._on_sim_setting_changed)
        self._settings.unbind(MotorSettings.simulate_rpm, self._on_sim_setting_changed)
        self._wakeup.set()
        self._sim_thread.join()

    # ------------------------------------------------------------------
    # Fall signal interface
    # ------------------------------------------------------------------

    def notify_fall(self) -> None:
        """External hardware fall signal.  No-op while simulate=True."""
        if self._settings.simulate:
            return
        self._fire_fall()

    def _fire_fall(self) -> None:
        """Record a fall unconditionally — used by both hardware and simulation."""
        now = monotonic()
        with self._fall_lock:
            if self._last_fall_time is not None:
                self._measured_period = now - self._last_fall_time
            self._last_fall_time = now

    # ------------------------------------------------------------------
    # Simulation thread
    # ------------------------------------------------------------------

    def _on_sim_setting_changed(self, _value) -> None:
        """Callback: wakes the sim loop on simulate/simulate_rpm change.
        Also clears fall history when simulate turns off so the first real
        hardware fall gets a clean measurement.
        """
        if not self._settings.simulate:
            with self._fall_lock:
                self._last_fall_time  = None
                self._measured_period = None
        self._wakeup.set()

    def _sim_loop(self) -> None:
        """Internal thread: fires synthetic falls at simulate_rpm cadence.

        last_fire tracks when the previous fall was fired.  On each wakeup
        (either natural timeout or a setting change via _wakeup) remaining time
        is recomputed from last_fire + period, so RPM changes take effect
        immediately without causing a spurious early fire.
        """
        last_fire: float = 0.0

        while self._running:
            self._wakeup.clear()
            rpm = self._settings.simulate_rpm if self._settings.simulate else 0.0
            if rpm <= 0.0:
                last_fire = 0.0
                self._wakeup.wait(0.1)  # timeout guards against missed wakeup
                continue
            remaining = last_fire + 60.0 / rpm - monotonic()
            if remaining > 0.0:
                self._wakeup.wait(remaining)
                continue
            self._fire_fall()
            last_fire = monotonic()

    # ------------------------------------------------------------------
    # Per-tick
    # ------------------------------------------------------------------

    def tick(self, target_rpm: float = 0.0) -> MotorState:
        """Advance and return the current motor state, echoing the commanded target_rpm."""
        offset   = self._settings.phase_offset
        simulate = self._settings.simulate

        low_speed_period = 60.0 / self._settings.low_speed_threshold
        latency_offset   = self._settings.latency_ms / 1000.0
        now              = monotonic()

        with self._fall_lock:
            last_fall_time  = self._last_fall_time
            measured_period = self._measured_period
            # Stall check and reset are done under the same lock so a concurrent
            # fall arriving between snapshot and reset cannot be wiped.
            if last_fall_time is not None and measured_period is not None:
                if now - last_fall_time > low_speed_period:
                    # No fall for longer than one low-speed revolution — motor has stalled.
                    self._last_fall_time  = None
                    self._measured_period = None
                    last_fall_time        = None
                    measured_period       = None

        if last_fall_time is not None and measured_period is not None:
            # Sub-tick precision: elapsed uses now captured before the lock (error < lock hold time, ~μs).
            # latency_ms pre-advances phase to compensate for signal transmission delay.
            # raw and latency are dimensionless revolution ratios (time/time); one × τ makes
            # them radians, then add the (radian) phase offset and wrap to [-π, π).
            elapsed = now - last_fall_time
            rpm     = 60.0 / measured_period
            raw     = elapsed / measured_period
            theta   = (min(raw, 1.0) + latency_offset / measured_period) * math.tau + offset
            playhead = (theta + math.pi) % math.tau - math.pi
        else:
            rpm      = 0.0
            playhead = 0.0

        if rpm == 0.0:
            motor_mode = MotorMode.STOPPED
        elif rpm >= self._settings.high_speed_threshold:
            motor_mode = MotorMode.HIGH_SPEED
        else:
            motor_mode = MotorMode.LOW_SPEED

        self._settings.measured_rpm = rpm
        self._settings.motor_mode   = motor_mode
        self._settings.playhead     = playhead
        return MotorState(playhead=playhead, mode=motor_mode, target_rpm=target_rpm)
