"""Tests for the Motor (raw measured phase + commanded mode) / Playhead (NCO) split."""

import math
import unittest
from time import monotonic

from apps.white_space.light.motor import MotorController, MotorSettings, MotorState, MotorMode, MotorSimMode
from apps.white_space.light.playhead import Playhead, PlayheadSettings

TAU = math.tau


def wrap(x: float) -> float:
    return (x + math.pi) % TAU - math.pi


def mstate(phase: float, locked: bool, rpm: float) -> MotorState:
    """A MotorState for the playhead: measured rpm when locked, commanded rpm otherwise."""
    return MotorState(
        phase=phase, locked=locked,
        measured_rpm=rpm if locked else 0.0,
        target_rpm=rpm,
    )


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

    def test_stall_unlocks_after_missed_revs(self) -> None:
        m = MotorController(MotorSettings())
        m._measured_period = 1.0
        m._last_fall_time = monotonic() - 5.0      # 5s > 3 revolutions (3 x 1s) → stalled
        self.assertFalse(m.tick().locked)
        m._measured_period = 1.0
        m._last_fall_time = monotonic() - 0.25     # within 3 revolutions → still locked
        self.assertTrue(m.tick().locked)


class SimTest(unittest.TestCase):
    def test_sim_rpm_from_mode(self) -> None:
        s = MotorSettings(); m = MotorController(s)
        s.simulate = MotorSimMode.LOW;     self.assertEqual(m._sim_rpm(), s.low_rpm)
        s.simulate = MotorSimMode.IDLE;    self.assertEqual(m._sim_rpm(), s.idle_rpm)
        s.simulate = MotorSimMode.HIGH;    self.assertEqual(m._sim_rpm(), s.high_rpm)
        s.simulate = MotorSimMode.STOPPED; self.assertEqual(m._sim_rpm(), 0.0)
        s.simulate = MotorSimMode.OFF;     self.assertEqual(m._sim_rpm(), 0.0)

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
        p = Playhead(PlayheadSettings()); p._phase = 2.0          # start 2 rad off
        mp = 0.0
        for _ in range(200):
            mp = next(gen)
            p.tick(dt, mstate(mp, True, rpm))
        self.assertAlmostEqual(wrap(p.phase - mp), 0.0, places=2)

    def test_free_run_advances_at_commanded_rpm(self) -> None:
        dt, rpm = 1 / 30, 72.0
        p = Playhead(PlayheadSettings())
        before = p.phase
        p.tick(dt, mstate(float("nan"), False, rpm))             # unlocked → uses target_rpm
        self.assertAlmostEqual(wrap(p.phase - before), rpm / 60.0 * TAU * dt, places=6)

    def test_continuity_across_rpm_step(self) -> None:
        dt = 1 / 60
        p = Playhead(PlayheadSettings())
        mp, prev, max_step = 0.0, p.phase, 0.0
        for i in range(300):
            rpm = 72.0 if i < 150 else 2000.0                    # instant jump (worst case)
            mp = wrap(mp + rpm / 60.0 * TAU * dt)
            p.tick(dt, mstate(mp, True, rpm))
            max_step = max(max_step, abs(wrap(p.phase - prev)))
            prev = p.phase
        self.assertLess(max_step, (2000.0 / 60.0 * TAU * dt) * 1.6)

    def test_offset_applied(self) -> None:
        p = Playhead(PlayheadSettings()); p._settings.offset = 0.5; p._phase = 1.0
        self.assertAlmostEqual(wrap(p.phase - 1.5), 0.0, places=6)

    def test_offset_constant_keeps_continuity(self) -> None:
        dt, rpm = 1 / 30, 72.0
        p = Playhead(PlayheadSettings()); p._settings.offset = -2.0736
        gen = _advancing(rpm, dt)
        prev = p.phase
        for _ in range(100):
            p.tick(dt, mstate(next(gen), True, rpm))
            self.assertLess(abs(wrap(p.phase - prev)), rpm / 60.0 * TAU * dt + 0.5)
            prev = p.phase


if __name__ == "__main__":
    unittest.main()
