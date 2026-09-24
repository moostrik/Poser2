"""Tests for ArmTravelExtractor — each arm joint's signed position along its travel, through the dead
zones at both ends."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Angles, AngleLandmark, ArmTravel, TravelElement
from modules.pose.nodes import ArmTravelExtractor, ArmTravelExtractorSettings

ZONE = 10.0                  # every zone (°) unless a test sets its own


def _frame(**arms: float) -> Frame:
    """A pose frame with the given arm angles (degrees, calibrated: 0 hanging / straight); every other angle NaN."""
    n = len(AngleLandmark)
    values = np.full(n, np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for name, degrees in arms.items():
        values[AngleLandmark[name]] = math.radians(degrees)
        scores[AngleLandmark[name]] = 1.0
    return Frame(track_id=0, cam_id=0, features={Angles: Angles(values, scores)})


def _settings(neutral: float = ZONE, far: float = ZONE) -> ArmTravelExtractorSettings:
    cfg = ArmTravelExtractorSettings()
    cfg.shoulder_neutral_dead_zone = cfg.elbow_neutral_dead_zone = neutral
    cfg.shoulder_raised_dead_zone = cfg.elbow_folded_dead_zone = far
    return cfg


class ArmTravelExtractorTest(unittest.TestCase):
    def _travel(self, frame: Frame, element: TravelElement = TravelElement.left_shoulder,
                neutral: float = ZONE, far: float = ZONE) -> float:
        return float(ArmTravelExtractor(_settings(neutral, far)).process(frame)[ArmTravel].values[element])

    def test_hanging_and_straight_read_zero(self) -> None:
        out = ArmTravelExtractor(_settings()).process(_frame(left_shoulder=0.0, right_shoulder=0.0, left_elbow=0.0, right_elbow=0.0))
        np.testing.assert_allclose(out[ArmTravel].values, 0.0)

    def test_inside_the_neutral_zone_is_still_zero(self) -> None:
        for degrees in (5.0, ZONE, -5.0):
            self.assertAlmostEqual(self._travel(_frame(left_shoulder=degrees)), 0.0, places=5, msg=f"{degrees}°")

    def test_between_the_zones_is_linear(self) -> None:
        self.assertAlmostEqual(self._travel(_frame(left_shoulder=20.0)), 10.0 / 160.0, places=5)
        self.assertAlmostEqual(self._travel(_frame(left_shoulder=90.0)), 0.5, places=5)

    def test_inside_the_far_zone_reads_one(self) -> None:
        for degrees in (180.0 - ZONE, 175.0, 180.0):
            self.assertAlmostEqual(self._travel(_frame(left_shoulder=degrees)), 1.0, places=5, msg=f"{degrees}°")

    def test_the_sign_is_kept(self) -> None:
        self.assertAlmostEqual(self._travel(_frame(left_shoulder=-90.0)), -0.5, places=5)
        self.assertAlmostEqual(self._travel(_frame(left_shoulder=-175.0)), -1.0, places=5)

    def test_the_elbow_has_its_own_zones(self) -> None:
        cfg = _settings()
        cfg.elbow_neutral_dead_zone, cfg.elbow_folded_dead_zone = 20.0, 40.0
        out = ArmTravelExtractor(cfg).process(_frame(left_shoulder=20.0, left_elbow=20.0, right_elbow=140.0))
        self.assertAlmostEqual(float(out[ArmTravel].values[TravelElement.left_shoulder]), 10.0 / 160.0, places=5)
        self.assertAlmostEqual(float(out[ArmTravel].values[TravelElement.left_elbow]), 0.0, places=5)
        self.assertAlmostEqual(float(out[ArmTravel].values[TravelElement.right_elbow]), 1.0, places=5)

    def test_zero_zones_are_the_angle_over_pi(self) -> None:
        for degrees in (0.0, 45.0, 90.0, 180.0):
            self.assertAlmostEqual(self._travel(_frame(left_shoulder=degrees), neutral=0.0, far=0.0), degrees / 180.0, places=5)

    def test_continuous_at_the_zone_edges(self) -> None:
        for edge in (ZONE, 180.0 - ZONE):
            below = self._travel(_frame(left_shoulder=edge - 0.1))
            above = self._travel(_frame(left_shoulder=edge + 0.1))
            self.assertLess(abs(above - below), 0.002, f"at {edge}°")

    def test_a_missing_joint_is_nan_with_score_zero(self) -> None:
        out = ArmTravelExtractor(_settings()).process(_frame(left_shoulder=90.0))[ArmTravel]
        self.assertTrue(math.isnan(float(out.values[TravelElement.right_shoulder])))
        self.assertEqual(float(out.scores[TravelElement.right_shoulder]), 0.0)
        self.assertEqual(float(out.scores[TravelElement.left_shoulder]), 1.0)

    def test_all_nan_leaves_feature_absent(self) -> None:
        out = ArmTravelExtractor(_settings()).process(_frame())
        self.assertNotIn(ArmTravel, out)
        self.assertTrue(np.all(np.isnan(out[ArmTravel].values)))


if __name__ == "__main__":
    unittest.main()
