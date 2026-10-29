"""Tests for HitSync — the hits' poses recorded at the hit and the streak of alike ones, on the group record."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from modules.pose import analytics, Age, Angles, AngleLandmark, ArmDeviation, GroupRecord

from apps.white_space.pose import Hit, HitSync, HitSyncSettings, HitStreak, NeutralWeightSettings

F = len(AngleLandmark)


def _angles(degrees: float) -> Angles:
    """Every joint at the same calibrated angle (degrees)."""
    values = np.full(F, math.radians(degrees), dtype=np.float32)
    return Angles(values, np.ones(F, dtype=np.float32))


class FakeFrame:
    """A player's LERP frame: their hit, pose, arm deviation and age (seconds; old enough by default)."""
    def __init__(self, hit: bool = False, angles: Angles | None = None, arms: float = 1.0,
                 age: float = 10.0) -> None:
        self.hit = hit
        self.angles = angles if angles is not None else _angles(0.0)
        self.arms = arms
        self.age = age

    def __getitem__(self, key):
        if key is Hit:
            return Hit.from_value(1.0 if self.hit else 0.0)
        if key is Angles:
            return self.angles
        if key is ArmDeviation:
            return ArmDeviation.from_value(self.arms) if not math.isnan(self.arms) else ArmDeviation.create_dummy()
        if key is Age:
            return Age.from_value(self.age) if not math.isnan(self.age) else Age.create_dummy()
        raise KeyError(key)


class HitSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = HitSyncSettings()
        posture = analytics.PostureSimilaritySettings()
        posture.angle_tolerance = 45.0
        self.neutral = NeutralWeightSettings()
        self.bars = 0.0
        self.sync = HitSync(self.settings, posture, self.neutral, lambda: self.bars)
        self.players = 3
        self.frames = {i: FakeFrame() for i in range(self.players)}

    def _tick(self, dbar: float = 0.0) -> HitStreak:
        self.bars += dbar
        return self.sync.record(self.frames, GroupRecord())[HitStreak]

    def hit(self, id: int, degrees: float = 60.0, arms: float = 1.0, age: float = 10.0) -> HitStreak:
        """The playhead crosses ``id`` this tick, a round divided by the players later than the last hit."""
        self.frames[id] = FakeFrame(hit=True, angles=_angles(degrees), arms=arms, age=age)
        streak = self._tick(dbar=1.0 / self.players)
        self.frames[id] = FakeFrame(angles=_angles(degrees), arms=arms, age=age)
        return streak

    def test_alike_hits_in_a_row_build_the_streak(self) -> None:
        self.assertEqual((self.hit(0).hits, self.hit(1, 70.0).hits), (1, 2))
        streak = self.hit(2, 80.0)
        self.assertEqual(streak.hits, 3)
        self.assertAlmostEqual(streak.distance, 20.0, places=3)       # the largest pair: 60° against 80°
        self.assertEqual(self.settings.hits, 3)
        self.assertAlmostEqual(self.settings.distance, 20.0, places=3)
        self.assertEqual(self._tick().hits, 3)                        # a tick without a hit keeps it

    def test_in_sync_is_the_postures_within_the_tolerance_for_every_pair(self) -> None:
        # Fully alike is the similarity reading 1: the distance within angle_tolerance (45° here). Just inside
        # passes, just outside does not, and every pair of the run must pass: 0° and 40° are alike, 40° and
        # 80° are alike, 0° and 80° are not.
        self.hit(0, 0.0)
        self.assertEqual(self.hit(1, 44.0).hits, 2)
        self.assertEqual(self.hit(2, 80.0).hits, 2)
        self.assertEqual(self.hit(0, 150.0).hits, 1)                  # not alike: the newest starts a new run

    def test_a_neutral_hit_is_alike_to_nothing_unless_the_weight_is_off(self) -> None:
        self.hit(0)
        self.assertEqual(self.hit(1, arms=0.9).hits, 1)               # a gate, not a ramp: nine tenths out is neutral
        self.assertEqual(self.hit(2, arms=math.nan).hits, 1)          # unseen arms are neutral
        self.neutral.enabled = False
        self.assertEqual(self.hit(0, arms=0.0).hits, 3)               # the posture alone decides: the neutral hits count too

    def test_hits_older_than_a_round_do_not_count(self) -> None:
        self.players = 2
        self.frames = {0: FakeFrame(), 1: FakeFrame()}
        self.hit(0)
        self.hit(1)
        self.assertEqual(self.hit(0).hits, 2)                         # three hits span a full bar: the oldest is out

    def test_a_young_poses_hit_is_not_in_the_streak(self) -> None:
        # A double or a false positive lives a moment: its hit neither extends nor restarts the streak; the
        # threshold is inclusive, a NaN age (the track's first frame) is 0, and at 0 every hit counts.
        self.players = 6
        self.hit(0)
        self.hit(1)
        self.assertEqual(self.hit(2, 150.0, age=1.9).hits, 2)
        self.assertEqual(self.hit(3, age=math.nan).hits, 2)
        self.assertEqual(self.hit(2, age=2.0).hits, 3)
        self.settings.min_age_seconds = 0.0
        self.assertEqual(self.hit(4, age=math.nan).hits, 4)

    def test_reset_forgets_the_hits(self) -> None:
        self.hit(0)
        self.hit(1)
        self.sync.reset()
        self.assertEqual(self._tick().hits, 0)


if __name__ == "__main__":
    unittest.main()
