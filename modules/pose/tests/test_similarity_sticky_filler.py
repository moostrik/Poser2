"""Tests for SimilarityStickyFiller — a present pair's gap is held on the analytics result; a departed player is not."""

import math
import unittest

from modules.pose.analytics import SimilarityResult, SimilarityStickyFiller, SimilarityStickyFillerSettings
from modules.pose.features import Similarity

from ._builders import scalar


def _rows(*pairs: dict[int, float]) -> SimilarityResult:
    """One row per given track (in order 0, 1, ...): ``{1: 0.7}`` puts 0.7 at slot 1, the rest NaN."""
    return SimilarityResult({tid: scalar(Similarity, values) for tid, values in enumerate(pairs)}, {0: "lead"})


class SimilarityStickyFillerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SimilarityStickyFillerSettings()
        self.filler = SimilarityStickyFiller(self.config)
        self.out: list[SimilarityResult] = []
        self.filler.add_similarity_callback(self.out.append)

    def _run(self, result: SimilarityResult) -> dict[int, Similarity]:
        self.filler.process(result)
        return self.out[-1].similarity

    def test_a_present_pairs_gap_is_held_with_score_zero(self) -> None:
        self._run(_rows({1: 0.7}, {0: 0.7}))
        rows = self._run(_rows({}, {}))                     # both present, no pair this frame
        self.assertAlmostEqual(rows[0][1], 0.7, places=6)
        self.assertAlmostEqual(rows[1][0], 0.7, places=6)
        self.assertEqual(rows[0].get_score(1), 0.0)

    def test_hold_scores_keeps_the_last_score(self) -> None:
        self.config.hold_scores = True
        self._run(_rows({1: 0.7}, {0: 0.7}))
        rows = self._run(_rows({}, {}))
        self.assertEqual(rows[0].get_score(1), 1.0)

    def test_a_departed_player_is_not_held(self) -> None:
        self._run(_rows({1: 0.7}, {0: 0.7}))
        rows = self._run(_rows({}))                         # track 1 has left
        self.assertTrue(math.isnan(rows[0][1]))
        self.assertEqual(rows[0].get_score(1), 0.0)
        self.assertNotIn(1, rows)

    def test_a_player_who_leaves_and_returns_starts_clean(self) -> None:
        self._run(_rows({1: 0.7}, {0: 0.7}))
        self._run(_rows({}))                                # track 1 gone
        rows = self._run(_rows({}, {}))                     # back, no pair yet
        self.assertTrue(math.isnan(rows[0][1]))
        self.assertTrue(math.isnan(rows[1][0]))

    def test_a_new_value_replaces_the_held_one(self) -> None:
        self._run(_rows({1: 0.7}, {0: 0.7}))
        self._run(_rows({}, {}))
        rows = self._run(_rows({1: 0.2}, {0: 0.2}))
        self.assertAlmostEqual(rows[0][1], 0.2, places=6)
        self.assertEqual(rows[0].get_score(1), 1.0)

    def test_the_self_slot_stays_nan(self) -> None:
        self._run(_rows({1: 0.7}, {0: 0.7}))
        rows = self._run(_rows({}, {}))
        self.assertTrue(math.isnan(rows[0][0]))

    def test_disabled_and_leader_pass_through(self) -> None:
        self.config.enabled = False
        result = _rows({1: 0.7}, {0: 0.7})
        self.filler.process(result)
        self.assertIs(self.out[-1], result)
        self.config.enabled = True
        self.filler.process(result)
        self.assertIs(self.out[-1].leader_score, result.leader_score)


if __name__ == "__main__":
    unittest.main()
