"""Tests for the Motor (raw measured phase + commanded mode) / Playhead (NCO) split."""

import math
import unittest
from time import monotonic

from apps.white_space.light.motor import (MotorController, MotorSettings, MotorMeasurement, MotorCommand,
                                          MotorMode)
from apps.white_space.light.playhead import Playhead, PlayheadSettings, _wrap_to_pi

TAU = math.tau


def wrap(x: float) -> float:
    return (x + math.pi) % TAU - math.pi


def mstate(phase: float, locked: bool, rpm: float,
           mode: MotorMode = MotorMode.BEAM, beam_rpm: float = 72.0) -> tuple[MotorMeasurement, MotorCommand]:
    """The (measurement, command) pair the playhead ticks on: `rpm` is the speed to sweep at —
    measured (with a valid phase) only when locked, otherwise the commanded target."""
    return (MotorMeasurement(phase=phase, locked=locked, measured_rpm=rpm if locked else 0.0),
            MotorCommand(mode=mode, target_rpm=rpm, beam_rpm=beam_rpm))


def running_playhead(settings: "PlayheadSettings | None" = None,
                     mode: MotorMode = MotorMode.BEAM) -> "Playhead":
    """A Playhead already in a rotating mode, so `.phase` is finite (not the stopped-NaN)."""
    p = Playhead(settings or PlayheadSettings())
    p._prev_mode = mode
    p._live = True
    return p


def _advancing(rpm: float, dt: float, start: float = 0.0):
    """Yield a phase advancing at `rpm`, like a steadily rotating light."""
    phase, rate = start, rpm / 60.0 * TAU
    while True:
        phase = wrap(phase + rate * dt)
        yield phase


class MotorTest(unittest.TestCase):
    def _locked(self, mode: MotorMode = MotorMode.BEAM, period: float = 1.0) -> MotorController:
        m = MotorController(MotorSettings())
        m.set_mode(mode)
        m._last_fall_time = monotonic() - 0.25
        m._measured_period = period
        return m

    def test_phase_locked_and_mode_commanded(self) -> None:
        m = self._locked(MotorMode.BEAM)
        st = m.tick()
        self.assertTrue(st.locked)
        self.assertTrue(-math.pi <= st.phase < math.pi)
        self.assertAlmostEqual(st.measured_rpm, 60.0, places=3)
        self.assertEqual(m.command.mode, MotorMode.BEAM)
        self.assertEqual(m.command.target_rpm, 72.0)   # beam_rpm default, derived from mode

    def test_mode_drives_target_rpm(self) -> None:
        self.assertEqual(self._locked(MotorMode.PROJECTION).command.target_rpm, 2000.0)
        self.assertEqual(self._locked(MotorMode.STOPPED).command.target_rpm, 0.0)

    def test_unlocked_when_no_falls(self) -> None:
        st = MotorController(MotorSettings()).tick()
        self.assertFalse(st.locked)
        self.assertTrue(math.isnan(st.phase))

    def test_boot_without_command_is_stopped(self) -> None:
        # There is no manual mode: until the machine (or debug) commands one, the motor
        # never spins — no command means STOPPED, so a boot can never start a fast spin.
        command = MotorController(MotorSettings()).command
        self.assertEqual(command.mode, MotorMode.STOPPED)
        self.assertEqual(command.target_rpm, 0.0)

    def test_duplicate_fall_does_not_divide_by_zero(self) -> None:
        # Two falls at the same instant (bouncing sensor / repeated packet) must not crash tick().
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)
        t = monotonic()
        m._fire_fall_at(t)
        m._fire_fall_at(t)              # simultaneous duplicate → no positive period recorded
        st = m.tick()                  # must not raise ZeroDivisionError
        self.assertFalse(st.locked)    # no valid measurement → not locked

    def test_duplicate_is_debounced_keeps_real_period(self) -> None:
        # A duplicate 32 ms after a real fall is ignored, so the measured period stays the real one
        # (without the debounce it would read ~1875 rpm for a full revolution).
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)
        t = monotonic()
        m._fire_fall_at(t)
        m._fire_fall_at(t + 0.5)        # real revolution: 0.5 s → 120 rpm
        m._fire_fall_at(t + 0.532)      # duplicate 32 ms later → debounced
        m._fire_fall_at(t + 1.0)        # next real revolution: 0.5 s after the REAL pulse
        st = m.tick()
        self.assertAlmostEqual(st.measured_rpm, 120.0, delta=2.0)   # real speed, not the duplicate spike

    def test_zero_period_does_not_lock(self) -> None:
        # Defensive: a degenerate 0 period is treated as no measurement, not a div-by-zero.
        m = MotorController(MotorSettings())
        m._last_fall_time = monotonic(); m._measured_period = 0.0
        self.assertFalse(m.tick().locked)

    def test_mode_is_commanded_immediately(self) -> None:
        # No stall/stop detection: the command is the commanded mode at once, even before any falls.
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)
        self.assertEqual(m.command.mode, MotorMode.BEAM)

    def test_long_gap_does_not_unlock(self) -> None:
        # No stall detection: a long silence never zeroes the measurement (fixes spin-down false-stops).
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)
        m._measured_period = 1.0                    # 60 rpm
        m._last_fall_time  = monotonic() - 30.0     # 30 s since the last fall — would have stalled before
        st = m.tick()
        self.assertTrue(st.locked)
        self.assertAlmostEqual(st.measured_rpm, 60.0)

    def test_high_no_falls_uses_command(self) -> None:
        # Commanded PROJECTION with no falls yet (cold start) → no measurement; the command stands on its own.
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.PROJECTION)
        st = m.tick()                                   # no falls
        self.assertFalse(st.locked)
        self.assertTrue(math.isnan(st.phase))
        self.assertEqual(st.measured_rpm, 0.0)
        self.assertEqual(m.command.mode, MotorMode.PROJECTION)
        self.assertEqual(m.command.target_rpm, 2000.0)  # commanded speed (no measurement yet)

    def test_high_ignores_stale_fall(self) -> None:
        # The motor sends no sync pulses above the ceiling, so PROJECTION ignores any leftover/stale fall
        # reading (otherwise a frozen pre-PROJECTION reading would drive the crossfade).
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.PROJECTION)
        m._last_fall_time = monotonic() - 0.001; m._measured_period = 1.0   # stale 60 rpm reading
        st = m.tick()
        self.assertFalse(st.locked)
        self.assertEqual(st.measured_rpm, 0.0)        # stale reading not used in PROJECTION
        self.assertTrue(math.isnan(st.phase))
        self.assertEqual(m.command.target_rpm, 2000.0)

    def test_above_ceiling_measurement_not_trusted(self) -> None:
        # Spinning down from PROJECTION: commanded BEAM but still physically fast → the >ceiling reading
        # (sensor can't keep up / phase aliases) is not trusted; only the command stands.
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)
        m._last_fall_time = monotonic() - 0.001; m._measured_period = 0.040   # 1500 rpm, above ceiling
        st = m.tick()
        self.assertFalse(st.locked)
        self.assertEqual(st.measured_rpm, 0.0)        # >ceiling reading discarded
        self.assertTrue(math.isnan(st.phase))
        self.assertEqual(m.command.target_rpm, 72.0)  # the command (beam_rpm)

    def test_no_falls_is_unlocked_with_the_command_standing(self) -> None:
        # With no measurement there is no 'stopped' state — the command stands.
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)                      # 72 rpm, below ceiling
        st = m.tick()                                  # no falls
        self.assertFalse(st.locked)
        self.assertEqual(m.command.mode, MotorMode.BEAM)
        self.assertEqual(m.command.target_rpm, 72.0)

    def test_commanded_stopped_is_idle(self) -> None:
        # Commanded STOPPED is the one deliberate 'off' path: no lock regardless of falls, rpm 0.
        m = self._locked(MotorMode.STOPPED)            # has falls, but commanded STOPPED
        st = m.tick()
        self.assertFalse(st.locked)
        self.assertEqual(m.command.mode, MotorMode.STOPPED)
        self.assertEqual(m.command.target_rpm, 0.0)


class CommandTest(unittest.TestCase):
    """`command` is the arbitration read at the moment it is asked for — the Conductor reads it
    once for the playhead (the command in force over the dt it advances) and once more after the
    state machine ran (the command the frame is sent with). `tick()` never touches it."""

    def test_reflects_set_mode_immediately(self) -> None:
        s = MotorSettings()
        m = MotorController(s)
        m.set_mode(MotorMode.PROJECTION)
        self.assertEqual(m.command, MotorCommand(MotorMode.PROJECTION, s.projection_rpm, s.beam_rpm))
        m.set_mode(MotorMode.BEAM)
        self.assertEqual(m.command, MotorCommand(MotorMode.BEAM, s.beam_rpm, s.beam_rpm))

    def test_a_command_issued_after_tick_is_visible_without_re_ticking(self) -> None:
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.PROJECTION)
        m.tick()
        m.set_mode(MotorMode.BEAM)                       # the machine commands BEAM after the tick
        self.assertEqual(m.command.mode, MotorMode.BEAM)

    def test_tick_leaves_the_command_alone(self) -> None:
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)
        before = m.command
        m.tick()
        self.assertEqual(m.command, before)

    def test_the_debug_override_outranks_the_machine(self) -> None:
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)
        m.set_debug_mode(MotorMode.PROJECTION)
        self.assertEqual(m.command.mode, MotorMode.PROJECTION)
        m.set_mode(MotorMode.STOPPED)
        self.assertEqual(m.command.mode, MotorMode.PROJECTION)
        m.set_debug_mode(None)
        self.assertEqual(m.command.mode, MotorMode.STOPPED)

    def test_active_mode_readback_follows_the_setters(self) -> None:
        s = MotorSettings()
        m = MotorController(s)
        m.set_mode(MotorMode.BEAM)
        self.assertEqual(s.active_mode, MotorMode.BEAM)
        m.set_debug_mode(MotorMode.PROJECTION)
        self.assertEqual(s.active_mode, MotorMode.PROJECTION)
        m.set_debug_mode(None)
        self.assertEqual(s.active_mode, MotorMode.BEAM)
        m.set_mode(None)
        self.assertEqual(s.active_mode, MotorMode.STOPPED)


class SimTest(unittest.TestCase):
    """The sim is a physical stand-in, never a decision-maker: it obeys the commanded
    mode (same arbitration as the real motor) and mimics the sensor's silence above
    the ceiling."""

    def test_sim_obeys_the_commanded_mode(self) -> None:
        # simulate on/off never changes the arbitration — the sim is not a mode source.
        s = MotorSettings()
        m = MotorController(s)
        s.simulate = True
        self.assertEqual(m._target_mode(), MotorMode.STOPPED)  # no command yet → no spin
        m.set_mode(MotorMode.PROJECTION)
        self.assertEqual(m._target_mode(), MotorMode.PROJECTION)    # machine command wins, sim or not
        s.simulate = False
        self.assertEqual(m._target_mode(), MotorMode.PROJECTION)

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
        s.simulate = True
        m.notify_fall()
        self.assertIsNone(m._last_fall_time)        # real falls ignored while simulating
        s.simulate = False
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
            p.tick(dt, *mstate(mp, True, rpm))
        self.assertAlmostEqual(wrap(p.phase - mp), 0.0, places=2)

    def test_stopped_holds_position(self) -> None:
        p = Playhead(PlayheadSettings()); p._internal = 1.0
        for _ in range(50):
            p.tick(1 / 60, *mstate(0.5, False, 0.0, mode=MotorMode.STOPPED))
        self.assertEqual(p._internal, 1.0)                       # internal sweep frozen — no advance
        self.assertTrue(math.isnan(p.phase))                     # stopped → NaN to consumers

    def test_high_free_runs_at_low_rpm(self) -> None:
        dt = 1 / 60
        p = running_playhead(mode=MotorMode.PROJECTION)
        before = p.phase
        # motor measured/target at 2000 but mode PROJECTION → playhead sweeps at beam_rpm, ignoring motor.phase
        p.tick(dt, *mstate(2.5, True, 2000.0, mode=MotorMode.PROJECTION, beam_rpm=72.0))
        self.assertAlmostEqual(wrap(p.phase - before), 72.0 / 60.0 * TAU * dt, places=6)

    def test_low_to_high_switch_is_seamless(self) -> None:
        dt = 1 / 60
        p = running_playhead()
        mp, prev, max_step = 0.0, p.phase, 0.0
        for i in range(300):
            if i < 150:                                          # BEAM: motor at 72, playhead follows
                mp = wrap(mp + 72.0 / 60.0 * TAU * dt)
                p.tick(dt, *mstate(mp, True, 72.0, mode=MotorMode.BEAM))
            else:                                                # PROJECTION: motor races at 2000, playhead ignores it
                mp = wrap(mp + 2000.0 / 60.0 * TAU * dt)
                p.tick(dt, *mstate(mp, True, 2000.0, mode=MotorMode.PROJECTION))
            max_step = max(max_step, abs(wrap(p.phase - prev)))
            prev = p.phase
        # never steps faster than the BEAM sweep — no jump at the switch, no speed-up to 2000
        self.assertLess(max_step, (72.0 / 60.0 * TAU * dt) * 1.6)

    def test_phase_nan_without_live_signal(self) -> None:
        dt = 1 / 60
        p = Playhead(PlayheadSettings())
        p.tick(dt, *mstate(0.5, False, 0.0, mode=MotorMode.STOPPED))
        self.assertTrue(math.isnan(p.phase))                    # stopped → NaN
        p.tick(dt, *mstate(0.5, False, 72.0, mode=MotorMode.BEAM))
        self.assertTrue(math.isnan(p.phase))                    # BEAM but unlocked (disconnected) → NaN
        p.tick(dt, *mstate(0.5, True, 72.0, mode=MotorMode.BEAM))
        self.assertFalse(math.isnan(p.phase))                   # locked → finite
        p.tick(dt, *mstate(0.5, False, 2000.0, mode=MotorMode.PROJECTION))
        self.assertFalse(math.isnan(p.phase))                   # PROJECTION free-runs content (unmeasurable) → finite

    def test_offset_applied(self) -> None:
        p = running_playhead(); p._settings.pulse_offset = 90.0; p._internal = 1.0   # 90° = π/2 rad
        self.assertAlmostEqual(wrap(p.phase - (1.0 + math.pi / 2)), 0.0, places=6)

    def test_offset_readout_is_degrees(self) -> None:
        p = running_playhead(); p._settings.pulse_offset = 90.0
        p.tick(1 / 30, *mstate(0.0, True, 72.0))
        self.assertTrue(0.0 <= p._settings.playhead <= 360.0)
        self.assertAlmostEqual(p._settings.playhead, math.degrees(p.phase) % 360.0, places=3)

    def test_offset_constant_keeps_continuity(self) -> None:
        dt, rpm = 1 / 30, 72.0
        p = running_playhead(); p._settings.pulse_offset = 360.0 - 118.8   # −0.33 turn
        gen = _advancing(rpm, dt)
        prev = p.phase
        for _ in range(100):
            p.tick(dt, *mstate(next(gen), True, rpm))
            self.assertLess(abs(wrap(p.phase - prev)), rpm / 60.0 * TAU * dt + 0.5)
            prev = p.phase


class ReacquireTest(unittest.TestCase):
    """PROJECTION → BEAM/IDLE re-acquire gate: after leaving PROJECTION the sweep keeps free-running at beam_rpm
    until the motor slows back to content speed, then re-locks onto the measured phase."""

    BEAM = MotorMode.BEAM
    PROJECTION = MotorMode.PROJECTION

    def _into_high(self, beam_rpm: float = 72.0) -> Playhead:
        """A playhead that has just been in PROJECTION (so leaving it arms the re-sync)."""
        p = running_playhead(mode=self.PROJECTION)
        p.tick(1 / 60, *mstate(float("nan"), False, 2000.0, mode=self.PROJECTION, beam_rpm=beam_rpm))
        return p

    def test_does_not_relock_while_motor_still_fast(self) -> None:
        dt = 1 / 60
        p = self._into_high()
        before = p._internal
        # Back to BEAM but the motor is still braking well above beam_rpm → keep free-running at beam_rpm,
        # do NOT advance at the (faster) measured rpm or snap toward the measured phase.
        p.tick(dt, *mstate(2.5, True, 150.0, mode=self.BEAM, beam_rpm=72.0))
        self.assertTrue(p._resyncing)
        self.assertFalse(math.isnan(p.phase))                          # re-syncing sweep is live
        self.assertAlmostEqual(wrap(p._internal - before), 72.0 / 60.0 * TAU * dt, places=6)

    def test_relocks_once_motor_reaches_content_speed(self) -> None:
        dt = 1 / 60
        p = self._into_high()
        p.tick(dt, *mstate(2.5, True, 150.0, mode=self.BEAM, beam_rpm=72.0))   # still fast → re-syncing
        self.assertTrue(p._resyncing)
        # Within tolerance of beam_rpm (72 × 1.05 = 75.6) → re-lock and resume tracking the measured phase.
        p.tick(dt, *mstate(2.5, True, 75.0, mode=self.BEAM, beam_rpm=72.0))
        self.assertFalse(p._resyncing)
        gen, mp = _advancing(72.0, dt, start=2.5), 2.5                 # now a steadily rotating motor
        for _ in range(200):                                           # tracking pulls onto the measured phase
            mp = next(gen)
            p.tick(dt, *mstate(mp, True, 72.0, mode=self.BEAM, beam_rpm=72.0))
        # Converges onto the measured phase (within ~1° — the speed EMA eases from the re-lock rpm to 72).
        self.assertAlmostEqual(wrap(p.phase - math.radians(p._settings.pulse_offset) - mp), 0.0, delta=0.02)

    def test_stale_content_speed_reading_does_not_relock(self) -> None:
        # Hardware spin-down: above the sensor ceiling the motor has no fresh falls and keeps reporting
        # its STALE pre-PROJECTION ≈beam_rpm measurement (locked, no stall detection). That must NOT re-lock
        # while the light is still physically spinning fast — otherwise the content races at the rpm the
        # sensor reports once falls resume (~ceiling). Re-lock only after a fresh fast reading settles.
        dt = 1 / 60
        p = self._into_high()
        before = p._internal
        # First BEAM ticks: stale content-speed reading, no fast measurement seen yet → stay re-syncing.
        for _ in range(3):
            p.tick(dt, *mstate(2.5, True, 72.0, mode=self.BEAM, beam_rpm=72.0))
            self.assertTrue(p._resyncing)
            self.assertFalse(p._seen_fast)
        self.assertAlmostEqual(wrap(p._internal - before), 3 * 72.0 / 60.0 * TAU * dt, places=6)  # free-ran at low
        # Falls resume as the light passes the ceiling (real fast measurement) → still re-syncing.
        p.tick(dt, *mstate(2.5, True, 180.0, mode=self.BEAM, beam_rpm=72.0))
        self.assertTrue(p._resyncing); self.assertTrue(p._seen_fast)
        # Now it has settled to content speed → re-lock.
        p.tick(dt, *mstate(2.5, True, 73.0, mode=self.BEAM, beam_rpm=72.0))
        self.assertFalse(p._resyncing)

    def test_resync_is_live_while_motor_unmeasurable(self) -> None:
        dt = 1 / 60
        p = self._into_high()
        before = p._internal
        # Just after PROJECTION the motor is still above the sensor ceiling → unlocked, no measurement.
        p.tick(dt, *mstate(float("nan"), False, 0.0, mode=self.BEAM, beam_rpm=72.0))
        self.assertTrue(p._resyncing)
        self.assertFalse(math.isnan(p.phase))                          # not NaN — a live free-running sweep
        self.assertAlmostEqual(wrap(p._internal - before), 72.0 / 60.0 * TAU * dt, places=6)

    def test_high_to_stopped_clears_gate_and_holds(self) -> None:
        dt = 1 / 60
        p = self._into_high()
        p._internal = 1.0
        p.tick(dt, *mstate(0.5, False, 0.0, mode=MotorMode.STOPPED, beam_rpm=72.0))
        self.assertFalse(p._resyncing)
        self.assertEqual(p._internal, 1.0)                             # held
        self.assertTrue(math.isnan(p.phase))


class SetModeTest(unittest.TestCase):
    """The state machine's command channel: set_mode() drives the motor while set;
    None relinquishes — no command means STOPPED (there is no manual mode)."""

    def test_command_drives_the_mode(self) -> None:
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.PROJECTION)
        self.assertEqual(m.command.mode, MotorMode.PROJECTION)

    def test_none_relinquishes_to_stopped(self) -> None:
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.PROJECTION)
        m.set_mode(None)
        self.assertEqual(m.command.mode, MotorMode.STOPPED)



class DebugOverrideTest(unittest.TestCase):
    """The debug rung outranks the machine; the auto-follow derives the mode from the
    selected debug layers."""

    def test_debug_mode_outranks_the_machine_command(self) -> None:
        m = MotorController(MotorSettings())
        m.set_mode(MotorMode.BEAM)                       # machine command
        m.set_debug_mode(MotorMode.PROJECTION)                # debug override wins
        self.assertEqual(m._target_mode(), MotorMode.PROJECTION)
        m.set_debug_mode(None)                          # debug off → machine again
        self.assertEqual(m._target_mode(), MotorMode.BEAM)
        m.set_mode(None)                                # machine relinquishes → STOPPED
        self.assertEqual(m._target_mode(), MotorMode.STOPPED)

    def test_auto_follow_derives_mode_from_selection(self) -> None:
        from types import SimpleNamespace
        from apps.white_space.light.conductor import _debug_motor_mode
        from apps.white_space.light import DebugLayer, LayerId
        layers = {LayerId.test_pose_waves: SimpleNamespace(MODE=MotorMode.PROJECTION),
                  LayerId.beam_playhead:    SimpleNamespace(MODE=MotorMode.BEAM)}
        self.assertEqual(_debug_motor_mode(DebugLayer.test_pose_waves, layers), MotorMode.PROJECTION)
        self.assertEqual(_debug_motor_mode(DebugLayer.beam_playhead, layers), MotorMode.BEAM)
        self.assertIsNone(_debug_motor_mode(DebugLayer.OFF, layers))         # debug disarmed

    def test_boot_failsafe_clears_debug(self) -> None:
        # A preset saved mid-debug (a projection layer selected) must never auto-derive PROJECTION at
        # power-on: the Conductor forces the select back to OFF at construction.
        from apps.white_space.light import Conductor, DebugLayer, LightSettings
        from apps.white_space.board import Board
        cfg = LightSettings()
        cfg.debug = DebugLayer.test_pose_waves
        Conductor(cfg, Board(), pose_stage=4)
        self.assertEqual(DebugLayer(int(cfg.debug)), DebugLayer.OFF)


class BarsTest(unittest.TestCase):
    """The playhead's monotonic bar counter (1 bar = 1 full playhead cycle): accumulates at
    the sweep's own rate, at the commanded content rate while unmeasured, and only STOPPED
    holds it."""

    def test_high_accumulates_at_content_rate(self) -> None:
        dt = 1 / 60
        p = running_playhead(mode=MotorMode.PROJECTION)
        for _ in range(60):
            p.tick(dt, *mstate(2.5, True, 2000.0, mode=MotorMode.PROJECTION, beam_rpm=72.0))
        self.assertAlmostEqual(p.bars, 72.0 / 60.0, places=6)   # 1 s at 72 rpm = 1.2 bars

    def test_locked_low_accumulates_at_measured_rate(self) -> None:
        dt = 1 / 60
        p = running_playhead()
        p.tick(dt, *mstate(0.0, True, 60.0, mode=MotorMode.BEAM))
        self.assertAlmostEqual(p.bars, 60.0 / 60.0 * dt, places=6)

    def test_unmeasured_low_advances_at_commanded_rate(self) -> None:
        dt = 1 / 60
        p = Playhead(PlayheadSettings())
        p.tick(dt, *mstate(float("nan"), False, 72.0, mode=MotorMode.BEAM))
        self.assertTrue(math.isnan(p.phase))                    # not live…
        self.assertAlmostEqual(p.bars, 72.0 / 60.0 * dt, places=6)   # …but bars never stall

    def test_stopped_holds(self) -> None:
        p = Playhead(PlayheadSettings())
        for _ in range(30):
            p.tick(1 / 60, *mstate(0.5, False, 0.0, mode=MotorMode.STOPPED))
        self.assertEqual(p.bars, 0.0)

    def test_monotonic_across_mode_changes(self) -> None:
        dt = 1 / 60
        p = running_playhead()
        prev = p.bars
        sequence = (
            [mstate(0.0, True, 72.0, mode=MotorMode.BEAM)] * 30
            + [mstate(2.5, True, 2000.0, mode=MotorMode.PROJECTION)] * 30
            + [mstate(float("nan"), False, 0.0, mode=MotorMode.BEAM)] * 30   # spin-down, unmeasurable
            + [mstate(0.0, True, 72.0, mode=MotorMode.BEAM)] * 30
        )
        for motor, command in sequence:
            p.tick(dt, motor, command)
            self.assertGreaterEqual(p.bars, prev)
            prev = p.bars
        self.assertGreater(p.bars, 0.0)


class RegimeSignalsTest(unittest.TestCase):
    """The playhead's mode-flip signals: `synced` (re-locked at BEAM — the stale-proof
    motor lock, which the S9/S10 exit anchors on) and
    `ring_formed` (PROJECTION + fall silence — the bar blurred into the ring)."""

    DT = 1 / 60

    def test_synced_only_while_tracking_at_low(self) -> None:
        p = running_playhead()
        p.tick(self.DT, *mstate(0.0, True, 72.0, mode=MotorMode.BEAM))
        self.assertTrue(p.synced)
        p.tick(self.DT, *mstate(2.5, True, 2000.0, mode=MotorMode.PROJECTION))
        self.assertFalse(p.synced)                       # PROJECTION free-runs — not tracking

    def test_ring_formed_needs_high_plus_fall_silence(self) -> None:
        p = running_playhead(mode=MotorMode.PROJECTION)
        motor, command = mstate(float("nan"), False, 2000.0, mode=MotorMode.PROJECTION)
        motor.fall_age = 0.1                             # falls still arriving (climbing below ceiling)
        p.tick(self.DT, motor, command)
        self.assertFalse(p.ring_formed)
        motor.fall_age = 1.0                             # silence beyond the window → ring formed
        p.tick(self.DT, motor, command)
        self.assertTrue(p.ring_formed)
        motor, command = mstate(0.0, True, 72.0, mode=MotorMode.BEAM)
        motor.fall_age = 999.0                           # silence in BEAM is never "ring formed"
        p.tick(self.DT, motor, command)
        self.assertFalse(p.ring_formed)

    def test_relock_is_gated_against_stale_readings(self) -> None:
        p = running_playhead(mode=MotorMode.PROJECTION)
        p.tick(self.DT, *mstate(float("nan"), False, 2000.0, mode=MotorMode.PROJECTION))
        # Back to BEAM: the stale pre-PROJECTION reading must NOT re-lock instantly (gate not passed).
        p.tick(self.DT, *mstate(2.5, True, 72.0, mode=MotorMode.BEAM))
        self.assertFalse(p.synced)
        # A fresh above-content reading (the real spin-down) passes the gate; still braking.
        p.tick(self.DT, *mstate(2.5, True, 180.0, mode=MotorMode.BEAM))
        self.assertFalse(p.synced)
        # Settled at content speed → re-lock.
        p.tick(self.DT, *mstate(2.5, True, 73.0, mode=MotorMode.BEAM))
        self.assertTrue(p.synced)


class SpeedSmoothingTest(unittest.TestCase):
    """The feed-forward sweep rate uses the *averaged* measured speed, so per-revolution rpm jitter
    does not wobble the playhead. `tracking=0` isolates the rate term from the phase correction."""

    def _settings(self, speed_smoothing: float) -> PlayheadSettings:
        s = PlayheadSettings(); s.tracking = 0.0; s.speed_smoothing = speed_smoothing
        return s

    def test_speed_smoothing_rejects_a_speed_spike(self) -> None:
        dt = 1 / 60
        p = running_playhead(self._settings(0.9))
        for _ in range(120):                                       # warm up: EMA settles at 72 rpm
            p.tick(dt, *mstate(0.0, True, 72.0))
        before = p._internal
        p.tick(dt, *mstate(0.0, True, 720.0))                       # one-tick 10× spike
        step = wrap(p._internal - before)
        self.assertLess(step, 72.0 / 60.0 * TAU * dt * 2.0)        # averaged — nowhere near the 10× raw step

    def test_no_smoothing_follows_raw_speed(self) -> None:
        dt = 1 / 60
        p = running_playhead(self._settings(0.0))
        for _ in range(10):
            p.tick(dt, *mstate(0.0, True, 72.0))
        before = p._internal
        p.tick(dt, *mstate(0.0, True, 720.0))                       # spike passes straight through at 0 smoothing
        self.assertAlmostEqual(wrap(p._internal - before), 720.0 / 60.0 * TAU * dt, places=3)

    def test_ema_seeds_on_reacquire(self) -> None:
        dt = 1 / 60
        p = running_playhead(self._settings(0.9))
        p.tick(dt, *mstate(float("nan"), False, 72.0, mode=MotorMode.BEAM))   # disconnected → hold, not tracking
        before = p._internal
        p.tick(dt, *mstate(0.0, True, 100.0, mode=MotorMode.BEAM))            # re-acquire → seed EMA to 100, not ramp from 0
        self.assertAlmostEqual(wrap(p._internal - before), 100.0 / 60.0 * TAU * dt, places=3)


if __name__ == "__main__":
    unittest.main()
