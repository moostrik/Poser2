"""Tests for the show StateMachine — the CSV transition graph in both modes,
participant debounce, bar-denominated durations, goto/hold, motor commands, and looks."""

import unittest
from types import SimpleNamespace

from apps.white_space.light import LayerId, LightSettings, MotorMode
from apps.white_space.statemachine import StateId, StateMachine, StateMachineSettings
from apps.white_space.statemachine import machine as machine_module

POSE_STAGE = 4


class FakeTracklet(SimpleNamespace):
    pass


def tracklet(active: bool = True) -> FakeTracklet:
    return FakeTracklet(is_active=active)


class FakeFrame:
    """frame[PlayheadOffset].value → the stored offset."""
    def __init__(self, offset: float) -> None:
        self._offset = offset

    def __getitem__(self, _key) -> SimpleNamespace:
        return SimpleNamespace(value=self._offset)


class FakeBoard:
    def __init__(self) -> None:
        self.tracklets: dict[int, FakeTracklet] = {}
        self.bars: float = 0.0
        self.synced: bool = False        # playhead re-locked at BEAM (motor lock)
        self.ring_formed: bool = False   # bar blurred into the ring (un-lock)
        self.frames: dict[int, FakeFrame] = {}

    def get_tracklets(self):
        return self.tracklets

    def get_playhead_signals(self):
        return SimpleNamespace(phase=float("nan"), bars=self.bars, synced=self.synced,
                               ring_formed=self.ring_formed)

    def get_frames(self, stage: int):
        assert stage == POSE_STAGE
        return self.frames


class FakeSimilarity:
    def __init__(self, value: float) -> None:
        self._value = value

    def overall_similarity(self) -> float:
        return self._value


class StateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.t = 1000.0
        self._orig_time = machine_module.time
        machine_module.time = SimpleNamespace(time=lambda: self.t)

        self.config = StateMachineSettings()
        self.light = LightSettings()
        self.board = FakeBoard()
        self.mixes: list = []
        self.resets: list = []
        self.motors: list = []
        self.emitted: list = []

        self.machine = StateMachine(
            self.config, self.light, board=self.board,
            set_mix=self.mixes.append,
            reset_layers=self.resets.append,
            set_motor=self.motors.append,
            pose_stage=POSE_STAGE,
        )
        self.machine.add_state_callback(self.emitted.append)

    def tearDown(self) -> None:
        self.machine.stop()
        machine_module.time = self._orig_time

    # -- helpers ------------------------------------------------------------

    def tick(self, dt: float = 0.1, dbar: float = 0.0) -> None:
        self.t += dt
        self.board.bars += dbar
        self.machine.update()

    def set_participants(self, n: int, settle: bool = True) -> None:
        """Set the raw tracklet count; when settle, tick past the debounce hold."""
        self.board.tracklets = {i: tracklet() for i in range(n)}
        if settle:
            self.tick()   # register the pending count
            self.tick(dt=self.config.count_hold_seconds + 0.01)

    @property
    def current(self) -> StateId:
        return StateId(int(self.config.current))

    def goto(self, state: StateId) -> None:
        self.config.manual.select = state
        self.machine._on_goto(True)
        self.tick()

    def boot(self) -> None:
        """Power on and wake into IDLE: the machine boots into OFF, the (fake) playhead
        locks, OFF hands over to the wake, and the wake's bar completes. The lock stays
        asserted afterwards — IDLE is locked at BEAM."""
        self.board.synced = True
        self.tick()                                             # OFF, locked → OFF_IDLE
        self.tick(dbar=self.config.off_idle_bars + 0.1)         # wake complete → IDLE
        self.assertEqual(self.current, StateId.IDLE)

    # -- startup ------------------------------------------------------------

    def test_first_tick_boots_into_off_and_commands_low(self) -> None:
        # Boot failsafe #1: the show starts dark at BEAM, never in a lit or PROJECTION state.
        self.tick()
        self.assertEqual(self.current, StateId.OFF)
        self.assertEqual(self.motors, [MotorMode.BEAM])
        self.assertEqual(self.mixes[-1], [])
        self.assertEqual(self.emitted[-1].stage, 0)

    def test_boot_waits_for_the_lock(self) -> None:
        # Booting is the same wake as a blackout release, gated on the physics: OFF holds
        # until the playhead has locked at BEAM, then fades in through OFF_IDLE.
        self.tick()
        self.tick(dt=999.0, dbar=50.0)                          # time and bars alone: still dark
        self.assertEqual(self.current, StateId.OFF)
        self.board.synced = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF_IDLE)
        self.assertEqual(self.emitted[-1].stage, int(StateId.OFF_IDLE))
        self.tick(dbar=self.config.off_idle_bars + 0.1)
        self.assertEqual(self.current, StateId.IDLE)

    def test_startup_ignores_persisted_select(self) -> None:
        # Failsafe: a preset saved mid-show (select = PLAY) must never boot into a
        # PROJECTION-motor state — the show always starts in OFF; select is only the goto target.
        self.config.manual.select = StateId.PLAY
        self.config.manual.hold = True          # isolate the boot state from conditions
        self.board.synced = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF)
        self.assertEqual(self.motors, [MotorMode.BEAM])

    # -- the stand-alone CSV graph -------------------------------------------

    def test_idle_to_idle_intro_on_participant(self) -> None:
        self.boot()
        self.set_participants(1)
        self.assertEqual(self.current, StateId.IDLE_INTRO)

    def test_idle_intro_to_intro_on_hit(self) -> None:
        self.boot()
        self.set_participants(1)
        self.board.frames = {0: FakeFrame(0.3)}   # playhead approaching
        self.tick()
        self.board.frames = {0: FakeFrame(-0.1)}  # just passed → hit
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)

    def test_idle_intro_winds_back_when_left_before_hit(self) -> None:
        self.boot()
        self.set_participants(1)
        self.assertEqual(self.current, StateId.IDLE_INTRO)
        self.set_participants(0)
        self.assertEqual(self.current, StateId.INTRO_IDLE)
        # Arrived from IDLE_INTRO (already bright): both channels ramp from 1.0 — no dip/blink
        self.assertEqual(self.mixes[-1], [(LayerId.searchlight, 1.0), (LayerId.sound_light, 1.0)])
        self.tick(dbar=self.config.intro_idle_bars + 0.1)
        self.assertEqual(self.current, StateId.IDLE)

    def test_wrap_flip_is_not_a_hit(self) -> None:
        self.boot()
        self.set_participants(1)
        self.board.frames = {0: FakeFrame(3.0)}    # far side, positive
        self.tick()
        self.board.frames = {0: FakeFrame(-3.0)}   # wrapped past ±π, not a pass
        self.tick()
        self.assertEqual(self.current, StateId.IDLE_INTRO)

    def _to_intro(self, participants: int = 3) -> None:
        self.boot()
        self.set_participants(participants)
        self.board.frames = {0: FakeFrame(0.3)}
        self.tick()
        self.board.frames = {0: FakeFrame(-0.1)}
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)
        self.board.frames = {}

    def test_enter_resets_are_explicit_and_targeted(self) -> None:
        # INTRO resets the flash layer; INTRO_PLAY resets the instrument (fresh patterns +
        # fill per cycle); PLAY inherits the running instrument — no reset on END → PLAY.
        self._to_intro(participants=3)
        self.assertIn([LayerId.playhead_flash], self.resets)
        self.machine.set_similarity(SimpleNamespace(similarity={
            0: FakeSimilarity(0.9), 1: FakeSimilarity(0.9), 2: FakeSimilarity(0.9)}))
        self.tick()
        self.assertIn([LayerId.pose_instrument], self.resets)
        self.resets.clear()
        self.tick(dt=self.config.spin_up_seconds + 0.1)   # INTRO_PLAY → PLAY
        self.assertEqual(self.current, StateId.PLAY)
        self.assertEqual(self.resets, [])                    # PLAY inherits, never resets

    def test_intro_to_intro_play_on_sync_and_through_to_play(self) -> None:
        self._to_intro(participants=3)
        self.machine.set_similarity(SimpleNamespace(similarity={
            0: FakeSimilarity(0.9), 1: FakeSimilarity(0.8), 2: FakeSimilarity(0.9)}))
        self.tick()
        self.assertEqual(self.current, StateId.INTRO_PLAY)
        self.assertEqual(self.motors[-1], MotorMode.PROJECTION)
        # Still physically lamps: the dim line holds unchanged from INTRO.
        self.assertEqual(self.mixes[-1], [(LayerId.searchlight, 0.4)])
        # The ring forms → hard mix: instrument (white full, blue easing) + playhead line.
        self.board.ring_formed = True
        self.tick()
        mix = dict(self.mixes[-1])
        white, blue = mix[LayerId.pose_instrument]
        self.assertEqual(white, 1.0)                          # hard
        self.assertLess(blue, 1.0)                            # easing in from the un-lock
        self.assertEqual(mix[LayerId.projection_playhead], 1.0)
        self.assertNotIn(LayerId.searchlight, mix)
        self.tick(dt=self.config.spin_up_seconds)
        self.assertEqual(self.current, StateId.PLAY)
        self.assertEqual(self.mixes[-1], [(LayerId.pose_instrument, 1.0),
                                          (LayerId.projection_playhead, 1.0)])

    def _to_play(self) -> None:
        self.test_intro_to_intro_play_on_sync_and_through_to_play()
        self.board.synced = False       # PROJECTION: the sweep free-runs, the BEAM lock is gone

    def test_play_to_end_and_end_idle(self) -> None:
        self._to_play()
        self.set_participants(0)
        self.assertEqual(self.current, StateId.END)
        # END advances on bars; with P == 0 it lands in END_IDLE
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_IDLE)
        self.assertEqual(self.motors[-1], MotorMode.BEAM)
        self.tick(dt=999.0)                            # time alone never exits a spin-down
        self.assertEqual(self.current, StateId.END_IDLE)
        self.board.synced = True                       # BEAM reacquired — but the fade is not done
        self.tick()
        self.assertEqual(self.current, StateId.END_IDLE)
        self.light.beam_layers.wind_down.progress = 1.0   # fade complete + lock → hand over
        self.tick()
        self.assertEqual(self.current, StateId.IDLE)

    def test_end_lands_in_end_intro_with_people_then_intro(self) -> None:
        self._to_play()
        self.set_participants(2)
        self.assertEqual(self.current, StateId.END)
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_INTRO)
        self.board.synced = True
        self.light.beam_layers.wind_down.progress = 1.0
        self.tick()                                    # fade complete + lock → hand over
        self.assertEqual(self.current, StateId.INTRO)

    def test_wind_down_states_hold_a_constant_mix_and_ride_the_layer(self) -> None:
        # S9/S10's fade lives in the wind_down layer: the mix is constant (the landing look
        # underneath the dying wall), the layer is reset on entry, stage_progress is the
        # layer's own readout, and the exit is the fade complete plus the motor lock.
        self._to_play()
        self.set_participants(2)
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_INTRO)
        self.assertIn([LayerId.wind_down], self.resets)          # fade restarted at the full wall
        self.assertEqual(self.mixes[-1], [(LayerId.wind_down, 1.0), (LayerId.searchlight, 0.4)])
        self.light.beam_layers.wind_down.progress = 0.5                # the layer's fade readout
        self.tick()
        self.assertAlmostEqual(self.emitted[-1].stage_progress, 0.5)
        self.tick(dt=999.0, dbar=5.0)                  # time and bars alone never exit
        self.assertEqual(self.current, StateId.END_INTRO)
        self.light.beam_layers.wind_down.progress = 1.0                # fade complete — but no lock yet
        self.tick(dbar=2.0)
        self.assertEqual(self.current, StateId.END_INTRO)
        self.board.synced = True                       # lock + fade complete → hand over
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)

    def test_end_idle_reveals_the_sound_visuals_on_the_fade(self) -> None:
        self._to_play()
        self.set_participants(0)
        self.assertEqual(self.current, StateId.END)
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_IDLE)
        self.light.beam_layers.wind_down.progress = 0.25
        self.tick()
        self.assertEqual(self.mixes[-1], [(LayerId.wind_down, 1.0), (LayerId.searchlight, 1.0),
                                          (LayerId.sound_light, 0.25)])

    def test_end_winds_back_to_play_never_jumps(self) -> None:
        self._to_play()
        self.set_participants(2)
        self.assertEqual(self.current, StateId.END)
        self.tick(dbar=self.config.end_bars / 2)   # half-way down
        self.set_participants(3)                    # people return
        self.assertEqual(self.current, StateId.END)   # no jump: winds back first
        self.tick(dbar=self.config.end_bars / 4)
        self.assertEqual(self.current, StateId.END)
        self.tick(dbar=self.config.end_bars)        # ramp reaches 0
        self.assertEqual(self.current, StateId.PLAY)

    # -- session mode ---------------------------------------------------------

    def test_session_intro_timeout(self) -> None:
        self.config.session.enabled = True
        self._to_intro(participants=1)
        self.tick(dt=self.config.session.intro_seconds + 1.0)
        self.assertEqual(self.current, StateId.INTRO_PLAY)

    def test_session_empty_room_never_spins_up(self) -> None:
        self.config.session.enabled = True
        self._to_intro(participants=1)
        self.set_participants(0)
        self.tick(dt=self.config.session.intro_seconds + 1.0)
        self.assertNotEqual(self.current, StateId.INTRO_PLAY)

    def test_session_play_timeout_and_end_only_winds_down(self) -> None:
        self.config.session.enabled = True
        self._to_play()
        self.assertEqual(self.current, StateId.PLAY)   # P == 3: no natural end
        self.tick(dt=self.config.session.play_seconds + 1.0)
        self.assertEqual(self.current, StateId.END)
        for _ in range(4):                                # P ≥ 3, but session: no wind-back
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_INTRO)

    # -- dev controls ----------------------------------------------------------

    def test_blackout_pins_off_from_anywhere(self) -> None:
        # Pinning blackout is OFF's entry door: pin → OFF immediately (dark strip,
        # /global/state 0), and OFF stays put while pinned. Dark and silent, but the
        # rotor keeps sweeping at BEAM so the playhead never unlocks.
        self._to_play()
        self.config.blackout = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF)
        self.assertEqual(self.motors[-1], MotorMode.BEAM)      # still sweeping — no re-acquire
        self.assertEqual(self.mixes[-1], [])                  # dark strip
        self.assertEqual(self.emitted[-1].stage, 0)           # /global/state 0 = off
        self.set_participants(3)                              # presence alone never leaves OFF
        self.tick(dt=999.0)
        self.assertEqual(self.current, StateId.OFF)

    def _to_off(self) -> None:
        self.config.blackout = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF)

    def test_unpin_wakes_through_the_wake_transition(self) -> None:
        # Leaving OFF is a transition, not a jump: the wake fades up over its bar and
        # lands in IDLE — even with people present (the graph re-introduces them).
        self._to_play()                                       # 3 participants present
        self._to_off()
        self.board.synced = True                              # spun down to BEAM and re-locked
        self.config.blackout = False
        self.tick()
        self.assertEqual(self.current, StateId.OFF_IDLE)
        self.assertEqual(self.motors[-1], MotorMode.BEAM)
        self.tick(dbar=self.config.off_idle_bars + 0.1)
        self.assertEqual(self.current, StateId.IDLE)

    def test_released_blackout_waits_for_the_lock(self) -> None:
        # The wake is gated on the lock for a blackout release too — it just already holds
        # in the normal case, since the rotor never stopped.
        self.boot()
        self._to_off()
        self.board.synced = False                             # e.g. a silent sensor
        self.config.blackout = False
        self.tick(dt=999.0, dbar=50.0)
        self.assertEqual(self.current, StateId.OFF)           # held dark
        self.board.synced = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF_IDLE)

    def test_wake_ramps_up_from_dark(self) -> None:
        self.boot()
        self._to_off()
        self.config.blackout = False
        self.tick()                                           # entered the wake at p ≈ 0
        for layer, weight in self.mixes[-1]:
            self.assertAlmostEqual(weight, 0.0, places=6)     # still dark on entry
        self.tick(dbar=self.config.off_idle_bars / 2)
        mix = dict(self.mixes[-1])
        self.assertGreater(mix[LayerId.searchlight], 0.0)    # searchlight fading up
        self.assertLess(mix[LayerId.searchlight], 1.0)
        self.assertEqual(mix[LayerId.searchlight], mix[LayerId.sound_light])

    def test_hit_mid_wake_goes_straight_to_intro(self) -> None:
        self.boot()
        self.set_participants(1)
        self._to_off()
        self.config.blackout = False
        self.tick()
        self.assertEqual(self.current, StateId.OFF_IDLE)
        self.board.frames = {0: FakeFrame(0.3)}               # playhead approaching
        self.tick(dbar=self.config.off_idle_bars / 4)
        self.assertEqual(self.current, StateId.OFF_IDLE)
        self.board.frames = {0: FakeFrame(-0.1)}              # swept past → hit, mid-fade
        self.tick(dbar=self.config.off_idle_bars / 4)
        self.assertEqual(self.current, StateId.INTRO)

    def test_repinning_mid_wake_snaps_back_to_off(self) -> None:
        self.boot()
        self._to_off()
        self.config.blackout = False
        self.tick()
        self.assertEqual(self.current, StateId.OFF_IDLE)
        self.config.blackout = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF)
        self.assertEqual(self.mixes[-1], [])

    def test_blackout_entry_beats_hold_exit_is_a_normal_condition(self) -> None:
        self.boot()
        self.config.manual.hold = True
        self.config.blackout = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF)           # entry wins over hold
        self.config.blackout = False
        self.tick()
        self.assertEqual(self.current, StateId.OFF)           # exit is condition-driven → held
        self.config.manual.hold = False
        self.tick()
        self.assertEqual(self.current, StateId.OFF_IDLE)      # released → normal exit

    def test_goto_off_behaves_like_any_goto(self) -> None:
        # OFF is a state like any other: goto + hold parks it dark; without hold its own
        # exit condition (blackout unpinned) fires immediately, like any bouncing goto.
        self.boot()
        self.config.manual.hold = True
        self.goto(StateId.OFF)
        self.assertEqual(self.current, StateId.OFF)
        self.assertEqual(self.motors[-1], MotorMode.BEAM)
        self.config.manual.hold = False
        self.tick()
        self.assertEqual(self.current, StateId.OFF_IDLE)      # unpinned → wakes right out

    def test_boot_failsafe_clears_blackout(self) -> None:
        # A preset saved with blackout pinned must never wake the installation dark.
        config = StateMachineSettings()
        config.blackout = True
        machine = StateMachine(config, LightSettings(), board=FakeBoard(),
                               set_mix=lambda _: None, reset_layers=lambda _: None,
                               set_motor=lambda _: None, pose_stage=POSE_STAGE)
        try:
            self.assertFalse(config.blackout)
        finally:
            machine.stop()

    def test_goto_jumps_and_commands_motor(self) -> None:
        self.boot()
        self.config.manual.hold = True          # park on the state (goto + hold workflow)
        self.goto(StateId.PLAY)
        self.assertEqual(self.current, StateId.PLAY)
        self.assertEqual(self.motors[-1], MotorMode.PROJECTION)

    def test_goto_without_hold_keeps_evaluating_conditions(self) -> None:
        self.boot()
        self.goto(StateId.PLAY)               # jumped with P == 0 → PLAY's own condition fires
        self.assertEqual(self.current, StateId.END)

    def test_hold_freezes_transitions(self) -> None:
        self.boot()
        self.config.manual.hold = True
        self.set_participants(2)
        self.assertEqual(self.current, StateId.IDLE)
        self.config.manual.hold = False
        self.tick()
        self.assertEqual(self.current, StateId.IDLE_INTRO)

    def test_boot_failsafe_clears_hold(self) -> None:
        # A preset saved mid-hold must never freeze the power-on show: the machine forces
        # manual.hold off at construction.
        config = StateMachineSettings()
        config.manual.hold = True
        machine = StateMachine(config, LightSettings(), board=FakeBoard(),
                               set_mix=lambda _: None, reset_layers=lambda _: None,
                               set_motor=lambda _: None, pose_stage=POSE_STAGE)
        try:
            self.assertFalse(config.manual.hold)
        finally:
            machine.stop()

    def test_sync_mode_counts_participants_in_sync(self) -> None:
        from apps.white_space.statemachine import SyncMode
        self.config.sync.mode = SyncMode.ALL
        self._to_intro(participants=4)
        # 3 of 4 in sync: enough for THREE, not for ALL
        self.machine.set_similarity(SimpleNamespace(similarity={
            0: FakeSimilarity(0.9), 1: FakeSimilarity(0.9),
            2: FakeSimilarity(0.9), 3: FakeSimilarity(0.1)}))
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)
        self.config.sync.mode = SyncMode.THREE
        self.tick()
        self.assertEqual(self.current, StateId.INTRO_PLAY)

    def test_participant_flicker_is_debounced(self) -> None:
        self.boot()
        self.board.tracklets = {0: tracklet()}
        self.tick()                                   # pending, not yet effective
        self.assertEqual(self.current, StateId.IDLE)
        self.board.tracklets = {}
        self.tick()                                   # flicker back before the hold expired
        self.tick(dt=self.config.count_hold_seconds + 0.1)
        self.assertEqual(self.current, StateId.IDLE)


if __name__ == "__main__":
    unittest.main()
