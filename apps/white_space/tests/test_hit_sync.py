"""Tests for HitSync — the hits' poses recorded at the crossing and the streak of alike ones, on the board."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from modules.board import HitStreak
from modules.pose.analytics import PostureSimilaritySettings
from modules.pose.features import Angles, AngleLandmark, ArmDeviation

from apps.white_space.light import LightSettings
from apps.white_space.pose import HitSync, HitSyncSettings, NeutralWeightSettings, PlayheadOffset

POSE_STAGE = 4
F = len(AngleLandmark)


def _angles(degrees: float) -> Angles:
    """Every joint at the same calibrated angle (degrees)."""
    values = np.full(F, math.radians(degrees), dtype=np.float32)
    return Angles(values, np.ones(F, dtype=np.float32))


class FakeFrame:
    """A player's LERP frame: their playhead offset, pose and arm deviation."""
    def __init__(self, offset: float = math.nan, angles: Angles | None = None, arms: float = 1.0) -> None:
        self.offset = offset
        self.angles = angles if angles is not None else _angles(0.0)
        self.arms = arms

    def __getitem__(self, key):
        if key is PlayheadOffset:
            return SimpleNamespace(value=self.offset)
        if key is Angles:
            return self.angles
        if key is ArmDeviation:
            return ArmDeviation.from_value(self.arms) if not math.isnan(self.arms) else ArmDeviation.create_dummy()
        raise KeyError(key)


class FakeBoard:
    def __init__(self) -> None:
        self.bars = 0.0
        self.frames: dict[int, FakeFrame] = {}
        self.streaks: list[HitStreak] = []

    def get_playhead_signals(self):
        return SimpleNamespace(phase=float("nan"), bars=self.bars, is_locked=True, is_projecting=False)

    def get_frames(self, stage: int):
        assert stage == POSE_STAGE
        return self.frames

    def set_hit_streak(self, streak: HitStreak) -> None:
        self.streaks.append(streak)


class HitSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HitSyncSettings()
        posture = PostureSimilaritySettings()
        posture.angle_tolerance = 45.0
        self.neutral = NeutralWeightSettings()
        self.board = FakeBoard()
        self.sync = HitSync(self.config, posture, self.neutral, LightSettings(),
                            board=self.board, pose_stage=POSE_STAGE)
        self.players = 3
        self.board.frames = {i: FakeFrame() for i in range(self.players)}

    def _tick(self, dbar: float = 0.0) -> HitStreak:
        self.board.bars += dbar
        self.sync.update()
        return self.board.streaks[-1]

    def hit(self, id: int, degrees: float = 60.0, arms: float = 1.0) -> HitStreak:
        """The playhead crosses ``id`` this tick (the closest tick, −0.1 rad at 72 rpm / 30 Hz), a round
        divided by the players later than the last hit; the pass ends on the next tick."""
        self.board.frames[id] = FakeFrame(offset=-0.1, angles=_angles(degrees), arms=arms)
        streak = self._tick(dbar=1.0 / self.players)
        self.board.frames[id] = FakeFrame(angles=_angles(degrees), arms=arms)
        return streak

    def test_alike_hits_in_a_row_build_the_streak(self) -> None:
        self.assertEqual(self.hit(0), HitStreak(hit=True, hits=1, distance=0.0))
        self.assertEqual(self.hit(1, 70.0).hits, 2)
        streak = self.hit(2, 80.0)
        self.assertEqual(streak.hits, 3)
        self.assertAlmostEqual(streak.distance, 20.0, places=3)       # the largest pair: 60° against 80°
        self.assertEqual(self.config.hits, 3)
        self.assertAlmostEqual(self.config.distance, 20.0, places=3)

    def test_a_tick_without_a_crossing_keeps_the_streak_and_clears_the_hit_flag(self) -> None:
        self.hit(0)
        self.hit(1)
        streak = self._tick()
        self.assertEqual(streak, HitStreak(hit=False, hits=2, distance=0.0))

    def test_in_sync_is_the_postures_within_the_tolerance(self) -> None:
        # Fully alike is the similarity reading 1: the distance within angle_tolerance (45° here). Just inside
        # passes, just outside does not. No setting decides this but the tolerance.
        self.hit(0, 0.0)
        self.assertEqual(self.hit(1, 44.0).hits, 2)
        self.hit(2, 0.0)
        self.assertEqual(self.hit(0, 46.0).hits, 1)

    def test_every_pair_in_the_run_must_be_alike(self) -> None:
        # 0° and 40° are alike, 40° and 80° are alike, 0° and 80° are not: the run is the last two.
        self.hit(0, 0.0)
        self.hit(1, 40.0)
        self.assertEqual(self.hit(2, 80.0).hits, 2)

    def test_a_different_pose_restarts_the_count_at_once(self) -> None:
        self.hit(0, 60.0)
        self.hit(1, 60.0)
        self.assertEqual(self.hit(2, 150.0).hits, 1)          # not alike: the newest starts a new run
        self.assertEqual(self.hit(0, 150.0).hits, 2)          # the next alike hit already counts 2

    def test_a_neutral_hit_restarts_the_count(self) -> None:
        self.hit(0)
        self.hit(1)
        self.assertEqual(self.hit(2, arms=0.0).hits, 1)      # the glass ping is alike to nothing
        self.assertEqual(self.hit(0).hits, 1)                 # and the run before it is gone

    def test_unseen_arms_count_as_neutral(self) -> None:
        self.hit(0)
        self.assertEqual(self.hit(1, arms=math.nan).hits, 1)

    def test_the_neutral_gate_is_fully_out_of_neutral(self) -> None:
        # A gate, not a ramp: arms nine tenths out do not count, fully out do.
        self.hit(0)
        self.assertEqual(self.hit(1, arms=0.9).hits, 1)
        self.assertEqual(self.hit(2, arms=1.0).hits, 1)      # the 0.9 hit before it is still in the way
        self.assertEqual(self.hit(0, arms=1.0).hits, 2)

    def test_with_the_neutral_weight_off_a_neutral_hit_counts_as_alike(self) -> None:
        # The same switch as the live Similarity's weight: off, the posture alone decides.
        self.neutral.enabled = False
        self.hit(0)
        self.assertEqual(self.hit(1, arms=0.0).hits, 2)

    def test_hits_older_than_a_round_do_not_count(self) -> None:
        self.players = 2
        self.board.frames = {0: FakeFrame(), 1: FakeFrame()}
        self.hit(0)
        self.hit(1)
        self.assertEqual(self.hit(0).hits, 2)                 # three hits span a full bar: the oldest is out

    def test_a_pass_fires_once(self) -> None:
        self.board.frames[0] = FakeFrame(offset=0.1)          # the closest tick, still approaching
        self.assertEqual(self._tick().hit, True)
        self.board.frames[0] = FakeFrame(offset=-0.15)        # the next tick of the same pass
        self.assertEqual(self._tick().hit, False)
        self.assertEqual(self.board.streaks[-1].hits, 1)

    def test_reset_forgets_the_hits(self) -> None:
        self.hit(0)
        self.hit(1)
        self.sync.reset()
        self.assertEqual(self._tick(), HitStreak())


if __name__ == "__main__":
    unittest.main()
