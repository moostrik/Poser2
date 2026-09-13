"""Tests for the light strip's azimuth overlay geometry (eye vs bbox-centre strip positions)."""

import math
import unittest

from modules.pose.frame import Frame
from modules.pose.features import Azimuth
from apps.white_space.render.layers.azimuth_marks import build_azimuth_marks, signed_strip_gap


def _frame(track_id: int, azimuth: float) -> Frame:
    return Frame(track_id=track_id, cam_id=0, features={Azimuth: Azimuth.from_value(azimuth)})


class BuildAzimuthMarksTest(unittest.TestCase):
    def test_both_azimuths_map_to_strip_positions(self) -> None:
        marks = build_azimuth_marks({0: _frame(0, math.pi / 2)}, {0: _frame(0, -math.pi / 2)})
        self.assertEqual(len(marks), 1)
        self.assertAlmostEqual(marks[0].eye_x, 0.25, places=5)
        self.assertAlmostEqual(marks[0].bbox_x, 0.75, places=5)

    def test_missing_side_is_nan(self) -> None:
        marks = build_azimuth_marks({0: _frame(0, 0.0)}, {})
        self.assertAlmostEqual(marks[0].eye_x, 0.0, places=5)
        self.assertTrue(math.isnan(marks[0].bbox_x))

    def test_nan_azimuth_is_nan(self) -> None:
        marks = build_azimuth_marks({0: _frame(0, math.nan)}, {0: _frame(0, 1.0)})
        self.assertTrue(math.isnan(marks[0].eye_x))

    def test_neither_azimuth_is_left_out(self) -> None:
        self.assertEqual(build_azimuth_marks({0: _frame(0, math.nan)}, {}), [])

    def test_one_mark_per_person_with_a_pose(self) -> None:
        frames = {0: _frame(0, 0.0), 1: _frame(1, 0.0)}
        self.assertEqual([m.track_id for m in build_azimuth_marks(frames, frames)], [0, 1])


class SignedStripGapTest(unittest.TestCase):
    def test_plain_gap(self) -> None:
        self.assertAlmostEqual(signed_strip_gap(0.2, 0.3), 0.1, places=9)
        self.assertAlmostEqual(signed_strip_gap(0.3, 0.2), -0.1, places=9)

    def test_across_the_join(self) -> None:
        self.assertAlmostEqual(signed_strip_gap(0.98, 0.02), 0.04, places=9)
        self.assertAlmostEqual(signed_strip_gap(0.02, 0.98), -0.04, places=9)


if __name__ == "__main__":
    unittest.main()
