"""Tests for the Motor (raw measured phase + commanded mode) / Playhead (NCO) split."""

import math
import unittest
from time import monotonic

from apps.white_space.light.motor import MotorController, MotorSettings, MotorState, MotorMode, MotorSimMode
from apps.white_space.light.playhead import Playhead, PlayheadSettings, _lerp_angle, _wrap_to_pi

TAU = math.tau


def wrap(x: float) -> float:
    return (x + math.pi) % TAU - math.pi


def mstate(phase: float, locked: bool, rpm: float,
           mode: MotorMode = MotorMode.LOW, low_rpm: float = 72.0) -> MotorState:
    """A MotorState for the playhead: `rpm` is the effective speed to sweep at; measured rpm and phase
    are valid only when locked."""
    return MotorState(
        phase=phase, locked=locked,
        measured_rpm=rpm if locked else 0.0, effective_rpm=rpm,
        target_rpm=rpm, mode=mode, low_rpm=low_rpm,
    )


def running_playhead(settings: "PlayheadSettings | None" = None,
                     mode: MotorMode = MotorMode.LOW) -> "Playhead":
    """A Playhead already in a rotating mode, so `.phase` is finite (not the stopped-NaN)."""
    p = Playhead(settings or PlayheadSettings())
    p._prev_mode = mode
    return p


def _advancing(rpm: float, dt: float, start: float = 0.0):
    """Yield a phase advancing at `rpm`, like a steadily rotating light."""
    phase, rate = start, rpm / 60.0 * TAU
    while True:
        phase = wrap(phase + rate * dt)
        yield phase


class MotorTest(unittest.TestCase):
    def _locked(self, mode: MotorMode = MotorMode.LOW, period: float = 1.0) -> MotorController:
        s = MotorSettings(); s.mode = mode
        m = MotorController(s)
        m._last_fall_time = monotonic() - 0.25
        m._measured_period = period
        return m

    def test_phase_locked_and_mode_commanded(self) -> None:
        st = self._locked(MotorMode.LOW).tick()
        self.assertTrue(st.locked)
        self.assertTrue(-math.pi <= st.phase < math.pi)
        self.assertAlmostEqual(st.measured_rpm, 60.0, places=3)
        self.assertEqual(st.mode, MotorMode.LOW)
        self.assertEqual(st.target_rpm, 72.0)          # low_rpm default, derived from mode

    def test_mode_drives_target_rpm(self) -> None:
        self.assertEqual(self._locked(MotorMode.IDLE).tick().target_rpm, 7.0)
        self.assertEqual(self._locked(MotorMode.HIGH).tick().target_rpm, 2000.0)
        self.assertEqual(self._locked(MotorMode.STOPPED).tick().target_rpm, 0.0)

    def test_unlocked_when_no_falls(self) -> None:
        st = MotorController(MotorSettings()).tick()
        self.assertFalse(st.locked)
        self.assertTrue(math.isnan(st.phase))

    def test_mode_is_commanded_immediately(self) -> None:
        # No stall/stop detection: the reported mode is the commanded mode at once, even before any falls.
        s = MotorSettings(); s.mode = MotorMode.LOW
        self.assertEqual(MotorController(s).tick().mode, MotorMode.LOW)

    def test_long_gap_does_not_unlock(self) -> None:
        # No stall detection: a long silence never zeroes the measurement (fixes spin-down false-stops).
        s = MotorSettings(); s.mode = MotorMode.LOW
        m = MotorController(s)
        m._measured_period = 1.0                    # 60 rpm
        m._last_fall_time  = monotonic() - 30.0     # 30 s since the last fall — would have stalled before
        st = m.tick()
        self.assertTrue(st.locked)
        self.assertAlmostEqual(st.effective_rpm, 60.0)

    def test_high_runs_above_sensor_ceiling(self) -> None:
        # Above the sensor ceiling no falls arrive, so the motor is unlocked even while spinning.
        # Commanded HIGH (2000 > ceiling) → trust the command: report HIGH and act on target_rpm.
        s = MotorSettings(); s.mode = MotorMode.HIGH
        st = MotorController(s).tick()                  # no falls → unlocked
        self.assertFalse(st.locked)
        self.assertTrue(math.isnan(st.phase))          # phase still unmeasurable
        self.assertEqual(st.mode, MotorMode.HIGH)      # but reported as running in HIGH
        self.assertEqual(st.effective_rpm, 2000.0)     # acts on the commanded speed
        self.assertEqual(st.measured_rpm, 0.0)         # measured stays honest (sensor silent)

    def test_effective_rpm_is_measured_when_locked(self) -> None:
        st = self._locked(MotorMode.LOW, period=1.0).tick()   # 60 rpm measured, below the ceiling
        self.assertTrue(st.locked)
        self.assertAlmostEqual(st.effective_rpm, st.measured_rpm, places=6)

    def test_no_falls_uses_commanded_speed(self) -> None:
        # With no measurement, the effective speed follows the command — there is no 'stopped' state.
        s = MotorSettings(); s.mode = MotorMode.LOW    # 72 rpm, below ceiling
        st = MotorController(s).tick()                 # no falls
        self.assertFalse(st.locked)
        self.assertEqual(st.mode, MotorMode.LOW)
        self.assertEqual(st.effective_rpm, 72.0)

    def test_commanded_stopped_is_idle(self) -> None:
        # Commanded STOPPED is the one deliberate 'off' path: effective speed 0 regardless of falls.
        m = self._locked(MotorMode.STOPPED)            # has falls, but commanded STOPPED
        st = m.tick()
        self.assertEqual(st.mode, MotorMode.STOPPED)
        self.assertEqual(st.effective_rpm, 0.0)
        self.assertFalse(st.locked)


class SimTest(unittest.TestCase):
    def test_sim_mode_selects_target_rpm(self) -> None:
        s = MotorSettings(); m = MotorController(s)
        for sim, rpm in [(MotorSimMode.LOW, s.low_rpm), (MotorSimMode.IDLE, s.idle_rpm),
                         (MotorSimMode.HIGH, s.high_rpm), (MotorSimMode.STOPPED, 0.0)]:
            s.simulate = sim
            self.assertEqual(m._target_rpm(m._target_mode()), rpm)

    def test_sim_mode_names_map_to_motor_modes(self) -> None:
        # _target_mode() does MotorMode[sim.name]; guard the two enums against drifting apart.
        for sim in MotorSimMode:
            if sim is not MotorSimMode.OFF:
                self.assertIn(sim.name, MotorMode.__members__)

    def test_simulate_overrides_active_mode(self) -> None:
        s = MotorSettings(); s.mode = MotorMode.LOW
        m = MotorController(s)
        m._last_fall_time = monotonic() - 0.1; m._measured_period = 0.5
        s.simulate = MotorSimMode.HIGH
        st = m.tick()
        self.assertEqual(st.mode, MotorMode.HIGH)          # sim overrides the commanded mode
        self.assertEqual(st.target_rpm, 2000.0)
        s.simulate = MotorSimMode.OFF
        st = m.tick()
        self.assertEqual(st.mode, MotorMode.LOW)           # commanded mode rules when not simulating
        self.assertEqual(st.target_rpm, 72.0)

    def test_ramp_toward_models_inertia(self) -> None:
        r = MotorController._ramp_toward
        self.assertEqual(r(0.0, 2000.0, 100.0), 100.0)     # spin up, capped by max_step
        self.assertEqual(r(1950.0, 2000.0, 100.0), 2000.0) # never overshoots the target
        self.assertEqual(r(2000.0, 0.0, 100.0), 1900.0)    # spin down
        self.assertEqual(r(50.0, 0.0, 100.0), 0.0)         # never undershoots
        # 0 → 2000 at the default 333 rpm/s reaches target in ~6 s
        rpm, accel, dt, t = 0.0, 333.0, 0.05, 0.0
        while rpm < 2000.0:
            rpm = r(rpm, 2000.0, accel * dt); t += dt
        self.assertAlmostEqual(t, 6.0, delta=0.3)

    def test_notify_fall_gated_while_simulating(self) -> None:
        s = MotorSettings(); m = MotorController(s)
        s.simulate = MotorSimMode.LOW
        m.notify_fall()
        self.assertIsNone(m._last_fall_time)        # real falls ignored while simulating
        s.simulate = MotorSimMode.OFF
        m.notify_fall()
        self.assertIsNotNone(m._last_fall_time)


class PlayheadNcoTest(unittest.TestCase):
    def test_locks_to_advancing_motor(self) -> None:
        dt, rpm = 1 / 30, 72.0
        gen = _advancing(rpm, dt)
        p = Playhead(PlayheadSettings()); p._internal = 2.0       # start 2 rad off
        mp = 0.0
        for _ in range(200):
            mp = next(gen)
            p.tick(dt, mstate(mp, True, rpm))
        self.assertAlmostEqual(wrap(p.phase - mp), 0.0, places=2)

    def test_stopped_holds_position(self) -> None:
        p = Playhead(PlayheadSettings()); p._internal = p._output = 1.0
        for _ in range(50):
            p.tick(1 / 60, mstate(0.5, False, 0.0, mode=MotorMode.STOPPED))
        self.assertEqual(p._output, 1.0)                         # internal sweep frozen — no advance
        self.assertTrue(math.isnan(p.phase))                     # stopped → NaN to consumers

    def test_high_free_runs_at_low_rpm(self) -> None:
        dt = 1 / 60
        p = running_playhead(mode=MotorMode.HIGH)
        before = p.phase
        # motor measured/target at 2000 but mode HIGH → playhead sweeps at low_rpm, ignoring motor.phase
        p.tick(dt, mstate(2.5, True, 2000.0, mode=MotorMode.HIGH, low_rpm=72.0))
        self.assertAlmostEqual(wrap(p.phase - before), 72.0 / 60.0 * TAU * dt, places=6)

    def test_low_to_high_switch_is_seamless(self) -> None:
        dt = 1 / 60
        p = running_playhead()
        mp, prev, max_step = 0.0, p.phase, 0.0
        for i in range(300):
            if i < 150:                                          # LOW: motor at 72, playhead follows
                mp = wrap(mp + 72.0 / 60.0 * TAU * dt)
                p.tick(dt, mstate(mp, True, 72.0, mode=MotorMode.LOW))
            else:                                                # HIGH: motor races at 2000, playhead ignores it
                mp = wrap(mp + 2000.0 / 60.0 * TAU * dt)
                p.tick(dt, mstate(mp, True, 2000.0, mode=MotorMode.HIGH))
            max_step = max(max_step, abs(wrap(p.phase - prev)))
            prev = p.phase
        # never steps faster than the LOW sweep — no jump at the switch, no speed-up to 2000
        self.assertLess(max_step, (72.0 / 60.0 * TAU * dt) * 1.6)

    def test_phase_nan_only_when_stopped(self) -> None:
        dt = 1 / 60
        p = Playhead(PlayheadSettings())
        p.tick(dt, mstate(0.5, False, 0.0, mode=MotorMode.STOPPED))
        self.assertTrue(math.isnan(p.phase))                    # stopped → NaN to consumers
        for mode in (MotorMode.IDLE, MotorMode.LOW, MotorMode.HIGH):
            p.tick(dt, mstate(0.5, True, 72.0, mode=mode))
            self.assertFalse(math.isnan(p.phase))               # rotating → finite

    def test_offset_applied(self) -> None:
        p = running_playhead(); p._settings.offset = 0.5; p._output = 1.0
        self.assertAlmostEqual(wrap(p.phase - 1.5), 0.0, places=6)

    def test_offset_constant_keeps_continuity(self) -> None:
        dt, rpm = 1 / 30, 72.0
        p = running_playhead(); p._settings.offset = -2.0736
        gen = _advancing(rpm, dt)
        prev = p.phase
        for _ in range(100):
            p.tick(dt, mstate(next(gen), True, rpm))
            self.assertLess(abs(wrap(p.phase - prev)), rpm / 60.0 * TAU * dt + 0.5)
            prev = p.phase


class SpinupRideTest(unittest.TestCase):
    """§11 — optional LOW→HIGH ride: output rides the accelerating motor until release_rpm,
    then eases back to the internal low sweep."""

    def _ride_playhead(self, **kw) -> Playhead:
        s = PlayheadSettings()
        s.high_follow, s.release_rpm, s.release_smooth = True, 1200.0, 0.5
        for k, v in kw.items():
            setattr(s, k, v)
        return Playhead(s)

    LOW = MotorMode.LOW
    HIGH = MotorMode.HIGH

    def test_lerp_angle_shortest_path(self) -> None:
        self.assertAlmostEqual(_lerp_angle(0.0, 1.0, 0.0), 0.0)
        self.assertAlmostEqual(_lerp_angle(0.0, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(_lerp_angle(0.0, 1.0, 0.5), 0.5)
        self.assertAlmostEqual(_lerp_angle(3.0, -3.0, 0.5), _wrap_to_pi(3.0 + 0.5 * _wrap_to_pi(-6.0)), places=6)

    def test_off_uses_internal_low_sweep(self) -> None:
        dt = 1 / 60
        p = self._ride_playhead(high_follow=False)
        p.tick(dt, mstate(0.0, True, 72.0, mode=self.LOW))
        before = p._internal
        p.tick(dt, mstate(1.0, True, 2000.0, mode=self.HIGH, low_rpm=72.0))
        self.assertEqual(p._output, p._internal)                       # no ride
        self.assertAlmostEqual(wrap(p._internal - before), 72.0 / 60.0 * TAU * dt, places=6)

    def test_rides_motor_below_release_rpm(self) -> None:
        dt = 1 / 60
        p = self._ride_playhead()
        p.tick(dt, mstate(0.0, True, 72.0, mode=self.LOW))             # LOW first → arms the edge
        p.tick(dt, mstate(2.5, True, 500.0, mode=self.HIGH, low_rpm=72.0))  # measured < release → ride
        self.assertTrue(p._riding)
        self.assertEqual(p._w, 1.0)
        self.assertAlmostEqual(p.phase, wrap(2.5), places=6)           # output == motor.phase

    def test_internal_stays_low_sweep_while_riding(self) -> None:
        dt = 1 / 60
        p = self._ride_playhead()
        p.tick(dt, mstate(0.0, True, 72.0, mode=self.LOW))
        before = p._internal
        p.tick(dt, mstate(2.0, True, 500.0, mode=self.HIGH, low_rpm=72.0))
        self.assertTrue(p._riding)
        self.assertAlmostEqual(wrap(p._internal - before), 72.0 / 60.0 * TAU * dt, places=6)

    def test_releases_and_eases_to_internal(self) -> None:
        dt = 1 / 60
        p = self._ride_playhead(release_smooth=0.1)
        p.tick(dt, mstate(0.0, True, 72.0, mode=self.LOW))
        p.tick(dt, mstate(1.0, True, 500.0, mode=self.HIGH, low_rpm=72.0))   # riding, w=1
        for _ in range(20):                                            # measured ≥ release → ease out (0.33s > 0.1)
            p.tick(dt, mstate(2.0, True, 1500.0, mode=self.HIGH, low_rpm=72.0))
        self.assertFalse(p._riding)
        self.assertEqual(p._w, 0.0)
        self.assertEqual(p._output, p._internal)                       # settled back on the internal sweep

    def test_ride_survives_transient_unlock(self) -> None:
        dt = 1 / 60
        p = self._ride_playhead()
        p.tick(dt, mstate(0.0, True, 72.0, mode=self.LOW))
        p.tick(dt, mstate(float("nan"), False, 0.0, mode=self.HIGH, low_rpm=72.0))  # entered HIGH still unlocked
        self.assertTrue(p._riding)                                     # armed, not cancelled
        self.assertEqual(p._output, p._internal)                       # no phase to ride yet
        p.tick(dt, mstate(2.0, True, 500.0, mode=self.HIGH, low_rpm=72.0))          # lock appears → ride
        self.assertTrue(p._riding)
        self.assertEqual(p._w, 1.0)
        self.assertAlmostEqual(p.phase, wrap(2.0), places=6)

    def test_ride_cancels_when_leaving_high(self) -> None:
        dt = 1 / 60
        p = self._ride_playhead()
        p.tick(dt, mstate(0.0, True, 72.0, mode=self.LOW))
        p.tick(dt, mstate(1.0, True, 500.0, mode=self.HIGH, low_rpm=72.0))   # riding
        p.tick(dt, mstate(0.5, True, 72.0, mode=self.LOW))             # back to LOW → cancel
        self.assertFalse(p._riding)
        self.assertEqual(p._output, p._internal)


if __name__ == "__main__":
    unittest.main()
