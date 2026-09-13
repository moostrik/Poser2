"""Tests for EyeAzimuthExtractor — shifting a pose's azimuth from the bbox centre to the eyes."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Azimuth, BBox, Points2D, PointLandmark
from modules.pose.nodes import EyeAzimuthExtractor
from modules.utils import Rect

# A fake projection: one image width spans K radians, offset per camera so cam_id is honoured.
K = 2.0


def _column_to_azimuth(cam_id: int, x: float) -> float:
    return cam_id * 10.0 + x * K


def _frame(azimuth: float = 0.5, rect: Rect | None = Rect(0.2, 0.1, 0.4, 0.8),
           left_eye_x: float | None = 0.5, right_eye_x: float | None = 0.5,
           cam_id: int = 1, score: float = 0.7) -> Frame:
    n = len(PointLandmark)
    values = np.full((n, 2), np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for lm, x in ((PointLandmark.left_eye, left_eye_x), (PointLandmark.right_eye, right_eye_x)):
        if x is not None:
            values[lm] = (x, 0.2)
            scores[lm] = 1.0
    features: dict = {Points2D: Points2D(values, scores), Azimuth: Azimuth.from_value(azimuth, score)}
    if rect is not None:
        features[BBox] = BBox.from_rect(rect)
    return Frame(track_id=0, cam_id=cam_id, features=features)


class EyeAzimuthExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = EyeAzimuthExtractor(_column_to_azimuth)

    def _azimuth(self, frame: Frame) -> float:
        return self.extractor.process(frame)[Azimuth].value

    def test_eyes_on_box_centre_leave_azimuth(self) -> None:
        self.assertAlmostEqual(self._azimuth(_frame()), 0.5, places=5)

    def test_shift_is_eye_midpoint_offset_through_projection(self) -> None:
        # Midpoint 0.7 of a 0.4-wide box is 0.08 image widths right of centre → 0.08 * K rad.
        out = self._azimuth(_frame(left_eye_x=0.6, right_eye_x=0.8))
        self.assertAlmostEqual(out, 0.5 + 0.2 * 0.4 * K, places=5)

    def test_one_valid_eye_is_used_alone(self) -> None:
        out = self._azimuth(_frame(left_eye_x=None, right_eye_x=0.25))
        self.assertAlmostEqual(out, 0.5 - 0.25 * 0.4 * K, places=5)

    def test_no_eyes_leaves_frame_unchanged(self) -> None:
        frame = _frame(left_eye_x=None, right_eye_x=None)
        self.assertIs(self.extractor.process(frame), frame)

    def test_nan_azimuth_leaves_frame_unchanged(self) -> None:
        frame = _frame(azimuth=math.nan, left_eye_x=0.9, right_eye_x=0.9)
        self.assertIs(self.extractor.process(frame), frame)

    def test_missing_box_leaves_frame_unchanged(self) -> None:
        frame = _frame(rect=None, left_eye_x=0.9, right_eye_x=0.9)
        self.assertIs(self.extractor.process(frame), frame)

    def test_shift_wraps_across_pi(self) -> None:
        out = self._azimuth(_frame(azimuth=math.pi - 0.05, left_eye_x=1.0, right_eye_x=1.0))
        self.assertAlmostEqual(out, -math.pi - 0.05 + 0.5 * 0.4 * K, places=5)

    def test_score_is_kept(self) -> None:
        out = self.extractor.process(_frame(left_eye_x=0.7, right_eye_x=0.7))
        self.assertAlmostEqual(out[Azimuth].score, 0.7, places=5)


if __name__ == "__main__":
    unittest.main()
