"""Tests for ShoulderDifferenceExtractor — the shoulders' signed travel difference, left less
right, from the travels' magnitudes."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import ArmTravel, TravelElement, ShoulderDifference
from modules.pose.nodes import ShoulderDifferenceExtractor


def _frame(left: float = float("nan"), right: float = float("nan"),
           left_score: float = 1.0, right_score: float = 1.0) -> Frame:
    """A pose frame with the two shoulders' travel; every other travel element NaN."""
    values = np.full(len(TravelElement), np.nan, dtype=np.float32)
    scores = np.zeros(len(TravelElement), dtype=np.float32)
    for element, value, score in ((TravelElement.left_shoulder, left, left_score),
                                  (TravelElement.right_shoulder, right, right_score)):
        if not math.isnan(value):
            values[element] = value
            scores[element] = score
    return Frame(track_id=0, cam_id=0, timestamp=0.0, features={ArmTravel: ArmTravel(values, scores)})


class ShoulderDifferenceExtractorTest(unittest.TestCase):
    def _difference(self, frame: Frame) -> ShoulderDifference:
        return ShoulderDifferenceExtractor().process(frame)[ShoulderDifference]

    def test_the_difference_is_signed_left_less_right(self) -> None:
        for left, right, expected in ((1.0, 0.0, 1.0), (0.0, 1.0, -1.0), (0.5, 0.5, 0.0)):
            self.assertAlmostEqual(self._difference(_frame(left, right)).value, expected, places=6,
                                   msg=f"{left}, {right}")

    def test_the_sign_of_a_travel_is_dropped(self) -> None:
        # The travel's sign is the side the limb passes; the difference reads the magnitudes.
        self.assertAlmostEqual(self._difference(_frame(-0.5, 0.25)).value, 0.25, places=6)

    def test_the_score_is_the_lesser_shoulders(self) -> None:
        self.assertAlmostEqual(self._difference(_frame(0.5, 0.25, left_score=1.0, right_score=0.4)).score,
                               0.4, places=6)

    def test_a_missing_shoulder_leaves_the_feature_absent(self) -> None:
        out = ShoulderDifferenceExtractor().process(_frame(left=0.5))
        self.assertNotIn(ShoulderDifference, out)
        self.assertTrue(math.isnan(out[ShoulderDifference].value))     # the frame contract's NaN dummy


if __name__ == "__main__":
    unittest.main()
