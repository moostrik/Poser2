"""Tests for NeutralWeight — a pair's similarity weighted by its less-moved member's arm deviation."""

import math
import unittest

import numpy as np

from modules.pose import features
from modules.pose.analytics import SimilarityResult
from modules.pose.features import ArmDeviation, Similarity
from modules.pose.frame import Frame

from apps.white_space.pose import NeutralWeight, NeutralWeightSettings

features.configure_features(4)
N = Similarity.length()


def _frame(id: int, arms: float | None) -> Frame:
    """A present person; ``arms`` is their ArmDeviation (0 neutral .. 1 fully out), None leaves it absent (NaN)."""
    feats = {} if arms is None else {ArmDeviation: ArmDeviation.from_value(arms)}
    return Frame(track_id=id, cam_id=0, features=feats)


def _row(**pairs: float) -> Similarity:
    """A Similarity row: ``_row(_1=0.9)`` puts 0.9 at slot 1 with score 1; other slots NaN, score 0."""
    values = np.full(N, np.nan, dtype=np.float32)
    scores = np.zeros(N, dtype=np.float32)
    for name, value in pairs.items():
        values[int(name[1:])] = value
        scores[int(name[1:])] = 1.0
    return Similarity(values, scores)


class NeutralWeightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = NeutralWeightSettings()
        self.weight = NeutralWeight(self.config)
        self.out: list[SimilarityResult] = []
        self.weight.add_similarity_callback(self.out.append)
        self.result = SimilarityResult(
            {0: _row(_1=0.9, _2=0.8), 1: _row(_0=0.9, _2=0.7), 2: _row(_0=0.8, _1=0.7)},
            {0: "lead"})

    def _run(self, frames: dict[int, Frame]) -> dict[int, Similarity]:
        self.weight.set_frames(frames)
        self.weight.process(self.result)
        return self.out[-1].similarity

    def test_everyone_fully_out_of_neutral_passes_the_rows_unchanged(self) -> None:
        rows = self._run({0: _frame(0, 1.0), 1: _frame(1, 1.0), 2: _frame(2, 1.0)})
        for id, row in self.result.similarity.items():
            np.testing.assert_allclose(rows[id].values, row.values, equal_nan=True)
            np.testing.assert_array_equal(rows[id].scores, row.scores)

    def test_a_neutral_person_zeroes_their_pairs_both_ways_and_keeps_the_rest(self) -> None:
        rows = self._run({0: _frame(0, 1.0), 1: _frame(1, 0.0), 2: _frame(2, 1.0)})
        self.assertEqual(rows[0][1], 0.0)
        self.assertEqual(rows[1][0], 0.0)
        self.assertEqual(rows[1][2], 0.0)
        self.assertEqual(rows[2][1], 0.0)
        self.assertAlmostEqual(rows[0][2], 0.8, places=6)
        self.assertAlmostEqual(rows[2][0], 0.8, places=6)
        self.assertEqual(rows[0].get_score(1), 1.0)          # a weighted pair is a real value, not missing data
        self.assertTrue(math.isnan(rows[0][0]))              # the self slot stays NaN

    def test_a_pair_reads_as_its_less_moved_member(self) -> None:
        rows = self._run({0: _frame(0, 1.0), 1: _frame(1, 0.5), 2: _frame(2, 0.25)})
        self.assertAlmostEqual(rows[0][1], 0.9 * 0.5, places=6)
        self.assertAlmostEqual(rows[1][0], 0.9 * 0.5, places=6)
        self.assertAlmostEqual(rows[0][2], 0.8 * 0.25, places=6)
        self.assertAlmostEqual(rows[1][2], 0.7 * 0.25, places=6)

    def test_unseen_arms_count_as_neutral(self) -> None:
        rows = self._run({0: _frame(0, 1.0), 1: _frame(1, None), 2: _frame(2, 1.0)})
        self.assertEqual(rows[0][1], 0.0)
        self.assertAlmostEqual(rows[0][2], 0.8, places=6)

    def test_disabled_passes_the_result_through(self) -> None:
        self.config.enabled = False
        self._run({0: _frame(0, 0.0), 1: _frame(1, 0.0), 2: _frame(2, 0.0)})
        self.assertIs(self.out[-1], self.result)

    def test_leader_scores_pass_through(self) -> None:
        self._run({0: _frame(0, 1.0), 1: _frame(1, 0.0), 2: _frame(2, 0.0)})
        self.assertIs(self.out[-1].leader_score, self.result.leader_score)


if __name__ == "__main__":
    unittest.main()
