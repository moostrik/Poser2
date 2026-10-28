"""Tests for AgeExtractor — how long a track has had a pose, restarting when the track is reset."""

import unittest

from modules.pose.frame import Frame
from modules.pose.features import Age
from modules.pose.nodes import AgeExtractor


def _frame(t: float) -> Frame:
    return Frame(track_id=0, cam_id=0, time_stamp=t, features={})


class AgeExtractorTest(unittest.TestCase):
    def test_age_counts_from_the_first_frame(self) -> None:
        ex = AgeExtractor()
        ex.process(_frame(10.0))
        self.assertAlmostEqual(ex.process(_frame(12.5))[Age].value, 2.5)

    def test_reset_starts_the_age_over(self) -> None:
        # A reset is a new person in the slot: they must not inherit the previous person's age.
        ex = AgeExtractor()
        ex.process(_frame(10.0))
        ex.process(_frame(20.0))
        ex.reset()
        ex.process(_frame(30.0))
        self.assertAlmostEqual(ex.process(_frame(31.0))[Age].value, 1.0)


if __name__ == "__main__":
    unittest.main()
