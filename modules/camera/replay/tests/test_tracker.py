"""Tests for the replay tracker: the assignment, the life of a track (NEW, TRACKED, LOST,
REMOVED), ids that stay with the person and are reused smallest first, and the age."""

import itertools
import unittest

import numpy as np

from modules.camera import TrackingStatus
from modules.camera.replay import Tracker, assign, overlap
from modules.utils import Rect


def person(x: float, y: float = 0.3, w: float = 0.1, h: float = 0.4) -> Rect:
    return Rect(x, y, w, h)


class AssignTest(unittest.TestCase):
    def test_matches_brute_force_on_small_matrices(self) -> None:
        rng = np.random.default_rng(1)
        for n, m in [(1, 1), (2, 3), (3, 2), (4, 4), (5, 3)]:
            for _ in range(20):
                cost = rng.random((n, m))
                pairs = assign(cost)
                k = min(n, m)
                self.assertEqual(len(pairs), k)
                total = sum(cost[r, c] for r, c in pairs)
                best = min(sum(cost[r, c] for r, c in zip(rows, columns))
                           for rows in itertools.permutations(range(n), k)
                           for columns in itertools.permutations(range(m), k))
                self.assertAlmostEqual(total, best, places=9)

    def test_empty(self) -> None:
        self.assertEqual(assign(np.zeros((0, 3))), [])
        self.assertEqual(assign(np.zeros((2, 0))), [])


class OverlapTest(unittest.TestCase):
    def test_iou(self) -> None:
        a = np.array([[0, 0, 10, 10]], dtype=float)
        b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
        np.testing.assert_allclose(overlap(a, b)[0], [1.0, 1 / 3, 0.0], atol=1e-6)


class TrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = Tracker(lost_frames=3)

    def statuses(self, tracks) -> list[tuple[int, TrackingStatus]]:
        return [(t.id, t.status) for t in tracks]

    def test_a_person_is_new_then_tracked_with_one_id_while_walking(self) -> None:
        out = self.tracker.update([person(0.1)])
        self.assertEqual(self.statuses(out), [(0, TrackingStatus.NEW)])
        self.assertEqual(out[0].age, 1)
        for i in range(1, 20):
            out = self.tracker.update([person(0.1 + 0.01 * i)])
            self.assertEqual(self.statuses(out), [(0, TrackingStatus.TRACKED)])
        self.assertEqual(out[0].age, 20)
        self.assertAlmostEqual(out[0].roi.x, 0.29, places=6)      # the measured box, not the filtered one

    def test_two_people_keep_their_ids_when_they_pass(self) -> None:
        left, right = 0.2, 0.6
        for i in range(15):
            out = self.tracker.update([person(left + 0.02 * i), person(right - 0.02 * i)])
        by_id = {t.id: t.roi.x for t in out}
        self.assertAlmostEqual(by_id[0], left + 0.28, places=6)      # id 0 is the one that started left
        self.assertAlmostEqual(by_id[1], right - 0.28, places=6)

    def test_someone_who_leaves_is_lost_then_removed_once_then_gone(self) -> None:
        self.tracker.update([person(0.1), person(0.5)])
        out = self.tracker.update([person(0.5)])
        self.assertEqual(self.statuses(out), [(0, TrackingStatus.LOST), (1, TrackingStatus.TRACKED)])
        np.testing.assert_allclose(tuple(out[0].roi), tuple(person(0.1)))     # the last box stands
        self.tracker.update([person(0.5)])
        out = self.tracker.update([person(0.5)])
        self.assertEqual(out[0].status, TrackingStatus.LOST)            # the third frame without: still lost
        out = self.tracker.update([person(0.5)])
        self.assertEqual(self.statuses(out), [(0, TrackingStatus.REMOVED), (1, TrackingStatus.TRACKED)])
        out = self.tracker.update([person(0.5)])
        self.assertEqual(self.statuses(out), [(1, TrackingStatus.TRACKED)])

    def test_a_lost_person_who_comes_back_in_time_keeps_the_id(self) -> None:
        self.tracker.update([person(0.1)])
        self.tracker.update([])
        self.tracker.update([])
        out = self.tracker.update([person(0.11)])
        self.assertEqual(self.statuses(out), [(0, TrackingStatus.TRACKED)])

    def test_a_new_person_takes_the_smallest_free_id(self) -> None:
        self.tracker.update([person(0.1), person(0.5), person(0.8)])
        for _ in range(5):
            self.tracker.update([person(0.1), person(0.8)])              # the middle one leaves and is removed
        out = self.tracker.update([person(0.1), person(0.8), person(0.4)])
        self.assertEqual(self.statuses(out), [(0, TrackingStatus.TRACKED), (1, TrackingStatus.NEW), (2, TrackingStatus.TRACKED)])

    def test_far_apart_boxes_are_not_matched(self) -> None:
        self.tracker.update([person(0.1)])
        out = self.tracker.update([person(0.8)])
        self.assertEqual(self.statuses(out), [(0, TrackingStatus.LOST), (1, TrackingStatus.NEW)])

    def test_nobody_is_fine(self) -> None:
        self.assertEqual(self.tracker.update([]), [])


if __name__ == '__main__':
    unittest.main()
