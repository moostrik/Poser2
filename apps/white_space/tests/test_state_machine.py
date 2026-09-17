"""Tests for the show StateMachine — the CSV transition graph in both modes,
player debounce, bar-denominated durations, goto/hold, motor commands, and looks. The hits and the
streak of alike ones come from the board (HitSync's, tested in test_hit_sync.py)."""

import unittest
from types import SimpleNamespace

from modules.board import HitStreak

from apps.white_space.light import LayerId, LightSettings, MotorMode
from apps.white_space.statemachine import StateId, StateMachine, StateMachineSettings
from apps.white_space.statemachine import machine as machine_module

POSE_STAGE = 4


class FakeFrame:
    """A pose frame is a present player; the machine reads nothing else from it."""


class FakeBoard:
    def __init__(self) -> None:
        self.bars: float = 0.0
        self.is_locked: bool = False       # the playhead lock at BEAM
        self.is_projecting: bool = False   # fast enough for the projection to show
        self.frames: dict[int, FakeFrame] = {}
        self.hit_streak: HitStreak = HitStreak()

    def get_playhead_signals(self):
        return SimpleNamespace(phase=float("nan"), bars=self.bars, is_locked=self.is_locked,
                               is_projecting=self.is_projecting)

    def get_frames(self, stage: int):
        assert stage == POSE_STAGE
        return self.frames

    def get_hit_streak(self) -> HitStreak:
        return self.hit_streak


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

    def set_players(self, n: int, settle: bool = True) -> None:
        """Set the raw pose count (no playhead offset yet); when settle, tick past the debounce hold."""
        self.board.frames = {i: FakeFrame() for i in range(n)}
        if settle:
            self.tick()   # register the pending count
            self.tick(dt=self.config.count_hold_seconds + 0.01)

    def hit(self, dbar: float = 0.0) -> None:
        """This tick the playhead crosses a player (HitSync's hit flag on the board), then the pass is over."""
        self.board.hit_streak = HitStreak(hit=True, hits=self.board.hit_streak.hits)
        self.tick(dbar=dbar)
        self.board.hit_streak = HitStreak(hit=False, hits=self.board.hit_streak.hits)

    def set_streak(self, hits: int) -> None:
        """HitSync's streak on the board: this many hits in a row struck alike poses."""
        self.board.hit_streak = HitStreak(hit=False, hits=hits)

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
        self.board.is_locked = True
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
        self.board.is_locked = True
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
        self.board.is_locked = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF)
        self.assertEqual(self.motors, [MotorMode.BEAM])

    # -- the stand-alone CSV graph -------------------------------------------

    def test_idle_to_idle_intro_on_player(self) -> None:
        self.boot()
        self.set_players(1)
        self.assertEqual(self.current, StateId.IDLE_INTRO)

    def test_idle_intro_to_intro_on_hit(self) -> None:
        self.boot()
        self.set_players(1)
        self.tick()
        self.assertEqual(self.current, StateId.IDLE_INTRO)   # present, not yet hit
        self.hit()
        self.assertEqual(self.current, StateId.INTRO)

    def test_idle_intro_winds_back_when_left_before_hit(self) -> None:
        self.boot()
        self.set_players(1)
        self.assertEqual(self.current, StateId.IDLE_INTRO)
        self.set_players(0)
        self.assertEqual(self.current, StateId.INTRO_IDLE)
        # Arrived from IDLE_INTRO (already bright): both channels ramp from 1.0 — no dip/blink
        self.assertEqual(self.mixes[-1], [(LayerId.beam_playhead, 1.0), (LayerId.beam_blue_sound, 1.0)])
        self.tick(dbar=self.config.intro_idle_bars + 0.1)
        self.assertEqual(self.current, StateId.IDLE)

    def _to_intro(self, players: int = 3) -> None:
        self.boot()
        self.set_players(players)
        self.hit()                                  # the playhead crosses someone: the intro begins
        self.assertEqual(self.current, StateId.INTRO)

    def test_enter_resets_are_explicit_and_targeted(self) -> None:
        # INTRO resets the flash layer; INTRO_PLAY resets the instrument (fresh patterns +
        # fill per cycle); PLAY inherits the running instrument — no reset on END → PLAY.
        self._to_intro(players=3)
        self.assertIn([LayerId.beam_flash], self.resets)
        self.set_streak(3)
        self.tick()
        self.assertIn([LayerId.pose_instrument], self.resets)
        self.resets.clear()
        self.tick(dt=self.config.spin_up_seconds + 0.1)   # INTRO_PLAY → PLAY
        self.assertEqual(self.current, StateId.PLAY)
        self.assertEqual(self.resets, [])                    # PLAY inherits, never resets

    def test_intro_to_intro_play_on_sync_and_through_to_play(self) -> None:
        self._to_intro(players=3)
        self.set_streak(3)                         # three alike hits in a row
        self.tick()
        self.assertEqual(self.current, StateId.INTRO_PLAY)
        self.assertEqual(self.motors[-1], MotorMode.PROJECTION)
        # Still physically lamps: the dim line holds unchanged from INTRO.
        self.assertEqual(self.mixes[-1], [(LayerId.beam_playhead, 0.4)])
        # Projecting → hard mix: instrument (white full, blue easing) + playhead line.
        self.board.is_projecting = True
        self.tick()
        mix = dict(self.mixes[-1])
        white, blue = mix[LayerId.pose_instrument]
        self.assertEqual(white, 1.0)                          # hard
        self.assertLess(blue, 1.0)                            # easing in from projecting
        self.assertEqual(mix[LayerId.projection_playhead], 1.0)
        self.assertNotIn(LayerId.beam_playhead, mix)
        self.tick(dt=self.config.spin_up_seconds)
        self.assertEqual(self.current, StateId.PLAY)
        self.assertEqual(self.mixes[-1], [(LayerId.pose_instrument, 1.0),
                                          (LayerId.projection_playhead, 1.0)])

    def _to_play(self) -> None:
        self.test_intro_to_intro_play_on_sync_and_through_to_play()
        self.board.is_locked = False       # PROJECTION: the sweep free-runs, the BEAM lock is gone

    def test_play_to_end_and_end_idle(self) -> None:
        self._to_play()
        self.set_players(0)
        self.assertEqual(self.current, StateId.END)
        # END advances on bars; with P == 0 it lands in END_IDLE
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_IDLE)
        self.assertEqual(self.motors[-1], MotorMode.BEAM)
        self.tick(dt=999.0)                            # time alone never exits a spin-down
        self.assertEqual(self.current, StateId.END_IDLE)
        self.board.is_locked = True                       # BEAM reacquired — but the fade is not done
        self.tick()
        self.assertEqual(self.current, StateId.END_IDLE)
        self.light.beam_layers.beam_wind_down.progress = 1.0   # fade complete + lock → hand over
        self.tick()
        self.assertEqual(self.current, StateId.IDLE)

    def test_end_lands_in_end_intro_with_people_then_intro(self) -> None:
        self._to_play()
        self.set_players(2)
        self.assertEqual(self.current, StateId.END)
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_INTRO)
        self.board.is_locked = True
        self.light.beam_layers.beam_wind_down.progress = 1.0
        self.tick()                                    # fade complete + lock → hand over
        self.assertEqual(self.current, StateId.INTRO)

    def test_wind_down_states_hold_a_constant_mix_and_ride_the_layer(self) -> None:
        # S9/S10's fade lives in the wind_down layer: the mix is constant (the landing look
        # underneath the dying wall), the layer is reset on entry, stage_progress is the
        # layer's own readout, and the exit is the fade complete plus the playhead lock.
        self._to_play()
        self.set_players(2)
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_INTRO)
        self.assertIn([LayerId.beam_wind_down], self.resets)          # fade restarted at the full wall
        self.assertEqual(self.mixes[-1], [(LayerId.beam_wind_down, 1.0), (LayerId.beam_playhead, 0.4)])
        self.light.beam_layers.beam_wind_down.progress = 0.5                # the layer's fade readout
        self.tick()
        self.assertAlmostEqual(self.emitted[-1].stage_progress, 0.5)
        self.tick(dt=999.0, dbar=5.0)                  # time and bars alone never exit
        self.assertEqual(self.current, StateId.END_INTRO)
        self.light.beam_layers.beam_wind_down.progress = 1.0                # fade complete — but no lock yet
        self.tick(dbar=2.0)
        self.assertEqual(self.current, StateId.END_INTRO)
        self.board.is_locked = True                       # lock + fade complete → hand over
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)

    def test_end_idle_reveals_the_sound_visuals_on_the_fade(self) -> None:
        self._to_play()
        self.set_players(0)
        self.assertEqual(self.current, StateId.END)
        for _ in range(4):
            self.tick(dbar=self.config.end_bars / 3)
        self.assertEqual(self.current, StateId.END_IDLE)
        self.light.beam_layers.beam_wind_down.progress = 0.25
        self.tick()
        self.assertEqual(self.mixes[-1], [(LayerId.beam_wind_down, 1.0), (LayerId.beam_playhead, 1.0),
                                          (LayerId.beam_blue_sound, 0.25)])

    def test_end_winds_back_to_play_never_jumps(self) -> None:
        self._to_play()
        self.set_players(2)
        self.assertEqual(self.current, StateId.END)
        self.tick(dbar=self.config.end_bars / 2)   # half-way down
        self.set_players(3)                    # people return
        self.assertEqual(self.current, StateId.END)   # no jump: winds back first
        self.tick(dbar=self.config.end_bars / 4)
        self.assertEqual(self.current, StateId.END)
        self.tick(dbar=self.config.end_bars)        # ramp reaches 0
        self.assertEqual(self.current, StateId.PLAY)

    # -- session mode ---------------------------------------------------------

    def test_session_intro_timeout(self) -> None:
        self.config.session.enabled = True
        self._to_intro(players=1)
        self.tick(dt=self.config.session.intro_seconds + 1.0)
        self.assertEqual(self.current, StateId.INTRO_PLAY)

    def test_session_empty_room_never_spins_up(self) -> None:
        self.config.session.enabled = True
        self._to_intro(players=1)
        self.set_players(0)
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

    def test_session_play_ignores_the_count(self) -> None:
        # A small session that timed out into the spin-up gets its full PLAY: only play_seconds ends it.
        self.config.session.enabled = True
        self._to_intro(players=1)
        self.tick(dt=self.config.session.intro_seconds + 1.0)
        self.assertEqual(self.current, StateId.INTRO_PLAY)
        self.tick(dt=self.config.spin_up_seconds + 0.1)
        self.assertEqual(self.current, StateId.PLAY)
        self.tick(dt=self.config.session.play_seconds / 2)
        self.assertEqual(self.current, StateId.PLAY)       # P == 1 does not end a session
        self.tick(dt=self.config.session.play_seconds)
        self.assertEqual(self.current, StateId.END)

    # -- dev controls ----------------------------------------------------------

    def test_blackout_pins_off_from_anywhere(self) -> None:
        # Pinning blackout is OFF's entry door: pin → OFF immediately (dark,
        # /global/state 0), and OFF stays put while pinned. Dark and silent, but the
        # rotor keeps sweeping at BEAM so the playhead never unlocks.
        self._to_play()
        self.config.blackout = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF)
        self.assertEqual(self.motors[-1], MotorMode.BEAM)      # still sweeping — no re-acquire
        self.assertEqual(self.mixes[-1], [])                  # empty mix: dark
        self.assertEqual(self.emitted[-1].stage, 0)           # /global/state 0 = off
        self.set_players(3)                              # presence alone never leaves OFF
        self.tick(dt=999.0)
        self.assertEqual(self.current, StateId.OFF)

    def _to_off(self) -> None:
        self.config.blackout = True
        self.tick()
        self.assertEqual(self.current, StateId.OFF)

    def test_unpin_wakes_through_the_wake_transition(self) -> None:
        # Leaving OFF is a transition, not a jump: the wake fades up over its bar and
        # lands in IDLE — even with people present (the graph re-introduces them).
        self._to_play()                                       # 3 players present
        self._to_off()
        self.board.is_locked = True                              # spun down to BEAM and re-locked
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
        self.board.is_locked = False                             # e.g. a silent sensor
        self.config.blackout = False
        self.tick(dt=999.0, dbar=50.0)
        self.assertEqual(self.current, StateId.OFF)           # held dark
        self.board.is_locked = True
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
        self.assertGreater(mix[LayerId.beam_playhead], 0.0)    # searchlight fading up
        self.assertLess(mix[LayerId.beam_playhead], 1.0)
        self.assertEqual(mix[LayerId.beam_playhead], mix[LayerId.beam_blue_sound])

    def test_hit_mid_wake_goes_straight_to_intro(self) -> None:
        self.boot()
        self.set_players(1)
        self._to_off()
        self.config.blackout = False
        self.tick()
        self.assertEqual(self.current, StateId.OFF_IDLE)
        self.tick(dbar=self.config.off_idle_bars / 4)
        self.assertEqual(self.current, StateId.OFF_IDLE)
        self.hit(dbar=self.config.off_idle_bars / 4)          # swept mid-fade → the intro begins
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
        self.set_players(2)
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

    def test_the_streak_builds_toward_min_players(self) -> None:
        # The players heard the same sound min_players times in a row: two alike hits hold INTRO, the third
        # spins up; the count is HitSync's, the machine only compares it.
        self._to_intro(players=3)
        self.set_streak(2)
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)
        self.set_streak(3)
        self.tick()
        self.assertEqual(self.current, StateId.INTRO_PLAY)

    def test_alike_hits_without_enough_players_present_hold_intro(self) -> None:
        # A streak can only be as long as the players in the room, but the presence gate is its own condition.
        self.config.min_players = 3
        self._to_intro(players=2)
        self.set_streak(3)
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)

    def test_min_players_of_two_runs_a_two_person_show(self) -> None:
        # The min_players is the show's size: two alike hits spin up, PLAY holds with two, and END
        # winds back to PLAY once two are back.
        self.config.min_players = 2
        self._to_intro(players=2)
        self.set_streak(2)
        self.tick()
        self.assertEqual(self.current, StateId.INTRO_PLAY)
        self.tick(dt=self.config.spin_up_seconds + 0.1)
        self.assertEqual(self.current, StateId.PLAY)
        self.board.is_locked = False
        self.set_players(1)
        self.assertEqual(self.current, StateId.END)
        self.tick(dbar=self.config.end_bars / 2)
        self.set_players(2)
        self.tick(dbar=self.config.end_bars)
        self.assertEqual(self.current, StateId.PLAY)

    def test_default_min_players_needs_three_people(self) -> None:
        # The default min_players of 3 keeps the three-person show: two alike hits with two present never spin up.
        self._to_intro(players=2)
        self.set_streak(2)
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)

    def test_the_streak_is_read_from_the_board_each_tick(self) -> None:
        # The machine holds no sync state of its own: the streak dropping on the board drops the context.
        self._to_intro(players=3)
        self.set_streak(2)
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)
        self.set_streak(1)
        self.tick()
        self.assertEqual(self.current, StateId.INTRO)
        self.set_streak(3)
        self.tick()
        self.assertEqual(self.current, StateId.INTRO_PLAY)

    def test_dim_level_is_the_intro_line(self) -> None:
        self.config.dim_level = 0.25
        self._to_intro(players=1)
        self.assertEqual(dict(self.mixes[-1])[LayerId.beam_playhead], 0.25)

    def test_player_flicker_is_debounced(self) -> None:
        self.boot()
        self.board.frames = {0: FakeFrame()}
        self.tick()                                   # pending, not yet effective
        self.assertEqual(self.current, StateId.IDLE)
        self.board.frames = {}
        self.tick()                                   # flicker back before the hold expired
        self.tick(dt=self.config.count_hold_seconds + 0.1)
        self.assertEqual(self.current, StateId.IDLE)


if __name__ == "__main__":
    unittest.main()
