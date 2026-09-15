"""The FOCUS layout's choice of poses: the longest present first, the dummy like any other."""

import math
import unittest

from modules.pose.features import Age
from modules.pose.frame import Frame

from apps.white_space.pose import dummy_id
from apps.white_space.render.focus import FOCUS_COLUMNS, focus_ids

PLAYERS = 4
DUMMY = dummy_id(PLAYERS)


def _frame(track_id: int, age: float | None) -> Frame:
    features = {} if age is None else {Age: Age.from_value(age)}
    return Frame(track_id, 0, time_stamp=0.0, features=features)


def _frames(**ages: float | None) -> dict[int, Frame]:
    """``t3=12.0`` is track 3 at 12 s; ``d`` is the dummy."""
    return {(DUMMY if name == 'd' else int(name[1:])): _frame(DUMMY if name == 'd' else int(name[1:]), age)
            for name, age in ages.items()}


class FocusIdsTest(unittest.TestCase):

    def test_longest_present_first(self) -> None:
        self.assertEqual(focus_ids(_frames(t0=2.0, t1=9.0, t2=5.0)), [1, 2, 0])

    def test_the_dummy_is_ordered_by_its_age_like_anyone(self) -> None:
        self.assertEqual(focus_ids(_frames(t0=2.0, d=6.0, t2=9.0)), [2, DUMMY, 0])

    def test_at_most_the_slots(self) -> None:
        ids = focus_ids(_frames(t0=1.0, t1=2.0, t2=3.0, t3=4.0, d=5.0))
        self.assertEqual(len(ids), FOCUS_COLUMNS)
        self.assertEqual(ids, [DUMMY, 3, 2])

    def test_a_nan_or_missing_age_counts_as_zero(self) -> None:
        self.assertEqual(focus_ids(_frames(t0=math.nan, t1=None, t2=0.5)), [2, 0, 1])

    def test_ties_by_id(self) -> None:
        self.assertEqual(focus_ids(_frames(t3=4.0, t1=4.0, t2=4.0, t0=4.0)), [0, 1, 2])

    def test_nothing_present(self) -> None:
        self.assertEqual(focus_ids({}), [])


if __name__ == '__main__':
    unittest.main()
