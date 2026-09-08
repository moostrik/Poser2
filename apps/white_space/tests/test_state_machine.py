"""Tests for the show StateMachine — the CSV transition graph in both regimes,
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
        self.synced: bool = False        # playhead re-locked at LOW (motor lock)
        self.ring_formed: bool = False   # bar blurred into the ring (un-lock)
        self.spin_down: float = 0.0      # gated normalized deceleration
        self.frames: dict[int, FakeFrame] = {}

    def get_tracklets(self):
        return self.tracklets

    def get_playhead_signals(self):
        return SimpleNamespace(phase=float("nan"), bars=self.bars, synced=self.synced,
                               ring_formed=self.ring_formed, spin_down=self.spin_down)

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
        self.config.select = state
        self.machine._on_goto(True)
        self.tick()

    # -- startup ------------------------------------------------------------

    def test_first_tick_enters_idle_and_commands_motor(self) -> None:
        self.tick()
        self.assertEqual(self.current, StateId.IDLE)
        self.assertEqual(self.motors, [MotorMode.LOW])
        self.assertEqual(self.mixes[-1], [(LayerId.playhead_lamp, 1.0)])
        self.assertEqual(self.emitted[-1].stage, int(StateId.IDLE))

    def test_startup_ignores_persisted_select(self) -> None:
        # Failsafe: a preset saved mid-show (select = PLAY) must never boot into a
        # HIGH-motor state — the show always starts in IDLE; select is only the goto target.
        self.config.select = StateId.PLAY
        self.config.hold = True                 # isolate the boot state from conditions
        self.tick()
        self.assertEqual(self.current, StateId.IDLE)
        self.assertEqual(self.motors, [MotorMode.LOW])

    # -- the stand-alone CSV graph -------------------------------------------

    def test_idle_to_idle_intro_on_participant(self) -> None:
        self.tick()
        self.set_participants(1)
        self.assertEqual(self.current, StateId.IDLE_INTRO)

    def test_idle_intro_to_intro_on_hit(self) -> None:
        self.tick()
        self.set_participants(1)
        self.board.frames = {0: FakeFrame(0.3)}   # playhead approaching
        self.tick()
        self.board.frames = {0: FakeFrame(-0.1)}  # just passed → hit
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)

    def test_idle_intro_winds_back_when_left_before_hit(self) -> None:
        self.tick()
        self.set_participants(1)
        self.assertEqual(self.current, StateId.IDLE_INTRO)
        self.set_participants(0)
        self.assertEqual(self.current, StateId.INTRO_IDLE)
        # Arrived from IDLE_INTRO (already bright): ramps from 1.0 — no visible dip
        self.assertEqual(self.mixes[-1], [(LayerId.playhead_lamp, 1.0)])
        self.tick(dbar=self.config.intro_idle_bars + 0.1)
        self.assertEqual(self.current, StateId.IDLE)

    def test_wrap_flip_is_not_a_hit(self) -> None:
        self.tick()
        self.set_participants(1)
        self.board.frames = {0: FakeFrame(3.0)}    # far side, positive
        self.tick()
        self.board.frames = {0: FakeFrame(-3.0)}   # wrapped past ±π, not a pass
        self.tick()
        self.assertEqual(self.current, StateId.IDLE_INTRO)

    def _to_intro(self, participants: int = 3) -> None:
        self.tick()
        self.set_participants(participants)
        self.board.frames = {0: FakeFrame(0.3)}
        self.tick()
        self.board.frames = {0: FakeFrame(-0.1)}
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)
        self.board.frames = {}

    def test_enter_resets_are_explicit_and_targeted(self) -> None:
        # INTRO resets the flash layer; INTRO_PLAY resets pose_waves (fresh instrument per
        # show cycle); PLAY inherits the running waves — no reset on the END → PLAY path.
        self._to_intro(participants=3)
        self.assertIn([LayerId.playhead_flash], self.resets)
        self.machine.set_similarity(SimpleNamespace(similarity={
            0: FakeSimilarity(0.9), 1: FakeSimilarity(0.9), 2: FakeSimilarity(0.9)}))
        self.tick()
        self.assertIn([LayerId.pose_waves], self.resets)
        self.resets.clear()
        self.tick(dt=self.config.intro_play_seconds + 0.1)   # INTRO_PLAY → PLAY
        self.assertEqual(self.current, StateId.PLAY)
        self.assertEqual(self.resets, [])                    # PLAY inherits, never resets

    def test_intro_to_intro_play_on_sync_and_through_to_play(self) -> None:
        self._to_intro(participants=3)
        self.machine.set_similarity(SimpleNamespace(similarity={
            0: FakeSimilarity(0.9), 1: FakeSimilarity(0.8), 2: FakeSimilarity(0.9)}))
        self.tick()
        self.assertEqual(self.current, StateId.INTRO_PLAY)
        self.assertEqual(self.motors[-1], MotorMode.HIGH)
        # mid spin-up the look blends lamp → pose_waves
        self.tick(dt=self.config.intro_play_seconds / 2)
        look = dict(self.mixes[-1])
        self.assertGreater(look[LayerId.pose_waves], 0.0)
        self.assertGreater(look[LayerId.playhead_lamp], 0.0)
        self.tick(dt=self.config.intro_play_seconds)
        self.assertEqual(self.current, StateId.PLAY)
        self.assertEqual(self.mixes[-1], [(LayerId.pose_waves, 1.0)])

    def _to_play(self) -> None:
        self.test_intro_to_intro_play_on_sync_and_through_to_play()

    def test_play_to_end_and_end_idle(self) -> None:
        self._to_play()
        self.set_participants(0)
        self.assertEqual(self.current, StateId.END)
        # END advances on bars; with P == 0 it lands in END_IDLE
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_IDLE)
        self.assertEqual(self.motors[-1], MotorMode.LOW)
        self.tick(dt=999.0)                            # time alone never exits a spin-down
        self.assertEqual(self.current, StateId.END_IDLE)
        self.board.synced = True                       # LOW reacquired — the physical exit
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
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)

    def test_spin_down_drives_the_fade_and_progress(self) -> None:
        # The S8 fade IS the deceleration: mix weights and stage_progress ride ctx.spin_down.
        self._to_play()
        self.set_participants(2)
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_INTRO)
        self.board.spin_down = 0.0                     # still above the ceiling: flood-era mix holds
        self.tick()
        self.assertEqual(dict(self.mixes[-1])[LayerId.pose_waves], 1.0)
        self.board.spin_down = 0.5                     # braking through the sensor range
        self.tick()
        mix = dict(self.mixes[-1])
        self.assertLess(mix[LayerId.pose_waves], 1.0)
        self.assertGreater(mix[LayerId.playhead_lamp], 0.0)
        self.assertAlmostEqual(self.emitted[-1].stage_progress, 0.5)
        self.board.spin_down = 1.0
        self.board.synced = True                       # re-lock: fade done and state done together
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)

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
        self.config.session = True
        self._to_intro(participants=1)
        self.tick(dt=self.config.intro_session_seconds + 1.0)
        self.assertEqual(self.current, StateId.INTRO_PLAY)

    def test_session_empty_room_never_spins_up(self) -> None:
        self.config.session = True
        self._to_intro(participants=1)
        self.set_participants(0)
        self.tick(dt=self.config.intro_session_seconds + 1.0)
        self.assertNotEqual(self.current, StateId.INTRO_PLAY)

    def test_session_play_timeout_and_end_only_winds_down(self) -> None:
        self.config.session = True
        self._to_play()
        self.assertEqual(self.current, StateId.PLAY)   # P == 3: no natural end
        self.tick(dt=self.config.play_session_seconds + 1.0)
        self.assertEqual(self.current, StateId.END)
        for _ in range(4):                                # P ≥ 3, but session: no wind-back
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_INTRO)

    # -- dev controls ----------------------------------------------------------

    def test_off_state_is_dark_stopped_and_goto_only(self) -> None:
        self.tick()
        self.goto(StateId.OFF)
        self.assertEqual(self.current, StateId.OFF)
        self.assertEqual(self.motors[-1], MotorMode.STOPPED)
        self.assertEqual(self.mixes[-1], [])                  # dark strip
        self.assertEqual(self.emitted[-1].stage, 0)           # /global/state 0 = off
        self.set_participants(3)                              # no condition leaves OFF
        self.tick(dt=999.0)
        self.assertEqual(self.current, StateId.OFF)
        self.set_participants(0)
        self.goto(StateId.IDLE)                               # operator resumes the show
        self.assertEqual(self.current, StateId.IDLE)
        self.assertEqual(self.motors[-1], MotorMode.LOW)

    def test_goto_jumps_and_commands_motor(self) -> None:
        self.tick()
        self.config.hold = True                 # park on the state (goto + hold workflow)
        self.goto(StateId.PLAY)
        self.assertEqual(self.current, StateId.PLAY)
        self.assertEqual(self.motors[-1], MotorMode.HIGH)

    def test_goto_without_hold_keeps_evaluating_conditions(self) -> None:
        self.tick()
        self.goto(StateId.PLAY)               # jumped with P == 0 → PLAY's own condition fires
        self.assertEqual(self.current, StateId.END)

    def test_hold_freezes_transitions(self) -> None:
        self.tick()
        self.config.hold = True
        self.set_participants(2)
        self.assertEqual(self.current, StateId.IDLE)
        self.config.hold = False
        self.tick()
        self.assertEqual(self.current, StateId.IDLE_INTRO)

    def test_disabled_relinquishes_motor_and_stops_transitions(self) -> None:
        self.tick()
        self.machine._on_enabled(False)
        self.config.enabled = False
        self.assertIsNone(self.motors[-1])
        self.set_participants(2)
        self.assertEqual(self.current, StateId.IDLE)

    def test_sync_mode_counts_participants_in_sync(self) -> None:
        from apps.white_space.statemachine import SyncMode
        self.config.sync_mode = SyncMode.ALL
        self._to_intro(participants=4)
        # 3 of 4 in sync: enough for THREE, not for ALL
        self.machine.set_similarity(SimpleNamespace(similarity={
            0: FakeSimilarity(0.9), 1: FakeSimilarity(0.9),
            2: FakeSimilarity(0.9), 3: FakeSimilarity(0.1)}))
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)
        self.config.sync_mode = SyncMode.THREE
        self.tick()
        self.assertEqual(self.current, StateId.INTRO_PLAY)

    def test_participant_flicker_is_debounced(self) -> None:
        self.tick()
        self.board.tracklets = {0: tracklet()}
        self.tick()                                   # pending, not yet effective
        self.assertEqual(self.current, StateId.IDLE)
        self.board.tracklets = {}
        self.tick()                                   # flicker back before the hold expired
        self.tick(dt=self.config.count_hold_seconds + 0.1)
        self.assertEqual(self.current, StateId.IDLE)


if __name__ == "__main__":
    unittest.main()
