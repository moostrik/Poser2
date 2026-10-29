"""Tests for SimilarityStickyFiller — a present pair's gap is held on the frames' rows; a departed player is not."""

import math
import unittest

from modules.pose.features import Similarity
from modules.pose.frame import FrameDict
from modules.pose.nodes import SimilarityStickyFiller, SimilarityStickyFillerSettings

from ._builders import frame, scalar


def _frames(*pairs: dict[int, float]) -> FrameDict:
    """One pose per given row (ids 0, 1, ...): ``{1: 0.7}`` puts 0.7 at slot 1, the rest NaN."""
    return {tid: frame(track_id=tid, features={Similarity: scalar(Similarity, values)}) for tid, values in enumerate(pairs)}


class SimilarityStickyFillerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SimilarityStickyFillerSettings()
        self.filler = SimilarityStickyFiller(self.settings)

    def _run(self, frames: FrameDict) -> dict[int, Similarity]:
        return {tid: f[Similarity] for tid, f in self.filler.process(frames).items()}

    def test_a_present_pairs_gap_is_held_with_score_zero(self) -> None:
        self._run(_frames({1: 0.7}, {0: 0.7}))
        rows = self._run(_frames({}, {}))                     # both present, no pair this frame
        self.assertAlmostEqual(rows[0][1], 0.7, places=6)
        self.assertAlmostEqual(rows[1][0], 0.7, places=6)
        self.assertEqual(rows[0].get_score(1), 0.0)

    def test_hold_scores_keeps_the_last_score(self) -> None:
        self.settings.hold_scores = True
        self._run(_frames({1: 0.7}, {0: 0.7}))
        rows = self._run(_frames({}, {}))
        self.assertEqual(rows[0].get_score(1), 1.0)

    def test_a_departed_player_is_not_held(self) -> None:
        self._run(_frames({1: 0.7}, {0: 0.7}))
        rows = self._run(_frames({}))                         # track 1 has left
        self.assertTrue(math.isnan(rows[0][1]))
        self.assertEqual(rows[0].get_score(1), 0.0)
        self.assertNotIn(1, rows)

    def test_a_player_who_leaves_and_returns_starts_clean(self) -> None:
        self._run(_frames({1: 0.7}, {0: 0.7}))
        self._run(_frames({}))                                # track 1 gone
        rows = self._run(_frames({}, {}))                     # back, no pair yet
        self.assertTrue(math.isnan(rows[0][1]))
        self.assertTrue(math.isnan(rows[1][0]))

    def test_a_new_value_replaces_the_held_one(self) -> None:
        self._run(_frames({1: 0.7}, {0: 0.7}))
        self._run(_frames({}, {}))
        rows = self._run(_frames({1: 0.2}, {0: 0.2}))
        self.assertAlmostEqual(rows[0][1], 0.2, places=6)
        self.assertEqual(rows[0].get_score(1), 1.0)

    def test_the_self_slot_stays_nan(self) -> None:
        self._run(_frames({1: 0.7}, {0: 0.7}))
        rows = self._run(_frames({}, {}))
        self.assertTrue(math.isnan(rows[0][0]))

    def test_disabled_passes_the_frames_through(self) -> None:
        self.settings.enabled = False
        frames = _frames({1: 0.7}, {0: 0.7})
        self.assertIs(self.filler.process(frames), frames)

    def test_reset_forgets_the_held_values(self) -> None:
        self._run(_frames({1: 0.7}, {0: 0.7}))
        self.filler.reset()
        rows = self._run(_frames({}, {}))
        self.assertTrue(math.isnan(rows[0][1]))


if __name__ == "__main__":
    unittest.main()
