"""The POSE layout's slots: the longest present first, the dummy like any other, no jumping."""

import math
import unittest

from modules.pose.features import Age
from modules.pose.frame import Frame

from apps.white_space.pose import dummy_id
from apps.white_space.render.pose_slots import POSE_SLOTS, longest_present, pose_slots

PLAYERS = 4
DUMMY = dummy_id(PLAYERS)


def _frame(track_id: int, age: float | None) -> Frame:
    features = {} if age is None else {Age: Age.from_value(age)}
    return Frame(track_id, 0, time_stamp=0.0, features=features)


def _frames(**ages: float | None) -> dict[int, Frame]:
    """``t3=12.0`` is track 3 at 12 s; ``d`` is the dummy."""
    return {(DUMMY if name == 'd' else int(name[1:])): _frame(DUMMY if name == 'd' else int(name[1:]), age)
            for name, age in ages.items()}


class LongestPresentTest(unittest.TestCase):

    def test_longest_present_first(self) -> None:
        self.assertEqual(longest_present(_frames(t0=2.0, t1=9.0, t2=5.0)), [1, 2, 0])

    def test_the_dummy_is_ordered_by_its_age_like_anyone(self) -> None:
        self.assertEqual(longest_present(_frames(t0=2.0, d=6.0, t2=9.0)), [2, DUMMY, 0])

    def test_at_most_the_slots(self) -> None:
        ids = longest_present(_frames(t0=1.0, t1=2.0, t2=3.0, t3=4.0, d=5.0))
        self.assertEqual(len(ids), POSE_SLOTS)
        self.assertEqual(ids, [DUMMY, 3, 2])

    def test_a_nan_or_missing_age_counts_as_zero(self) -> None:
        self.assertEqual(longest_present(_frames(t0=math.nan, t1=None, t2=0.5)), [2, 0, 1])

    def test_ties_by_id(self) -> None:
        self.assertEqual(longest_present(_frames(t3=4.0, t1=4.0, t2=4.0, t0=4.0)), [0, 1, 2])

    def test_nothing_present(self) -> None:
        self.assertEqual(longest_present({}), [])


class PoseSlotsTest(unittest.TestCase):

    def test_empty_slots_fill_longest_first(self) -> None:
        self.assertEqual(pose_slots([None, None, None], _frames(t0=2.0, t1=9.0, t2=5.0)), [1, 2, 0])

    def test_a_pose_keeps_its_slot_when_a_neighbour_leaves(self) -> None:
        slots = pose_slots([None, None, None], _frames(t0=2.0, t1=9.0, t2=5.0))     # [1, 2, 0]
        slots = pose_slots(slots, _frames(t0=3.0, t2=6.0))                          # 1 left
        self.assertEqual(slots, [None, 2, 0])

    def test_a_newcomer_takes_the_freed_slot(self) -> None:
        slots = pose_slots([None, 2, 0], _frames(t0=3.0, t2=6.0, t3=0.5))
        self.assertEqual(slots, [3, 2, 0])

    def test_a_pose_outlived_by_others_loses_its_slot(self) -> None:
        # 1 has been there longer than 3 all along; once present it takes 3's slot, the others keep theirs
        slots = pose_slots([3, 2, 0], _frames(t0=4.0, t1=5.0, t2=7.0, t3=1.5))
        self.assertEqual(slots, [1, 2, 0])

    def test_a_pose_stays_while_it_stays_among_the_longest(self) -> None:
        slots = [3, 2, 0]
        for _ in range(3):
            slots = pose_slots(slots, _frames(t0=4.0, t2=7.0, t3=1.5, t4=0.1))
        self.assertEqual(slots, [3, 2, 0])

    def test_fewer_poses_than_slots(self) -> None:
        self.assertEqual(pose_slots([None, None, None], _frames(d=1.0)), [DUMMY, None, None])
        self.assertEqual(pose_slots([DUMMY, None, None], {}), [None, None, None])


if __name__ == '__main__':
    unittest.main()
