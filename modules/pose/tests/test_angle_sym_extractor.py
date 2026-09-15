"""Tests for AngleSymExtractor — signed left-minus-right per pair, the limbs as wholes."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Angles, AngleLandmark, AngleSymmetry, SymmetryElement
from modules.pose.nodes import AngleSymExtractor, LegDeviationExtractorSettings


def _frame(**joints: float) -> Frame:
    """A pose frame with the given angles (radians); every other angle NaN."""
    n = len(AngleLandmark)
    values = np.full(n, np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    for name, angle in joints.items():
        values[AngleLandmark[name]] = angle
        scores[AngleLandmark[name]] = 1.0
    return Frame(track_id=0, cam_id=0, features={Angles: Angles(values, scores)})


def _settings() -> LegDeviationExtractorSettings:
    cfg = LegDeviationExtractorSettings()
    cfg.hip_degrees = 60.0
    cfg.knee_degrees = 90.0
    return cfg


class AngleSymExtractorTest(unittest.TestCase):
    def _sym(self, frame: Frame) -> AngleSymmetry:
        return AngleSymExtractor(_settings()).process(frame)[AngleSymmetry]

    def test_a_mirror_symmetric_pose_is_zero_everywhere(self) -> None:
        sym = self._sym(_frame(left_shoulder=0.7, right_shoulder=0.7, left_elbow=-0.2, right_elbow=-0.2,
                               left_hip=0.1, right_hip=0.1, left_knee=0.0, right_knee=0.0))
        np.testing.assert_allclose(sym.values, 0.0, atol=1e-6)
        np.testing.assert_array_equal(sym.scores, 1.0)

    def test_left_arm_higher_is_positive_on_the_shoulder_and_the_arms(self) -> None:
        sym = self._sym(_frame(left_shoulder=math.pi / 2, right_shoulder=0.0, left_elbow=0.0, right_elbow=0.0))
        self.assertAlmostEqual(sym[SymmetryElement.shoulder], 0.5, places=5)
        self.assertAlmostEqual(sym[SymmetryElement.elbow], 0.0, places=5)
        self.assertAlmostEqual(sym[SymmetryElement.arms], 0.25, places=5)
        self.assertLess(self._sym(_frame(left_shoulder=0.0, right_shoulder=math.pi / 2))[SymmetryElement.shoulder], 0.0)

    def test_the_difference_wraps_past_pi(self) -> None:
        # Straight up reads as −0.82π on one side and +1.18π would on the other: 0.1π apart, not 1.9π.
        sym = self._sym(_frame(left_shoulder=-0.82 * math.pi, right_shoulder=0.92 * math.pi))
        self.assertAlmostEqual(sym[SymmetryElement.shoulder], 0.26, places=4)

    def test_legs_are_weighted_as_the_leg_deviation(self) -> None:
        hip = self._sym(_frame(left_hip=math.pi / 6, right_hip=0.0, left_knee=0.0, right_knee=0.0))
        knee = self._sym(_frame(left_hip=0.0, right_hip=0.0, left_knee=math.pi / 6, right_knee=0.0))
        self.assertAlmostEqual(hip[SymmetryElement.legs], 0.25, places=5)      # π/6 over π/3, halved
        self.assertAlmostEqual(knee[SymmetryElement.legs], 1.0 / 6.0, places=5)  # π/6 over π/2, halved
        self.assertGreater(hip[SymmetryElement.legs], knee[SymmetryElement.legs])

    def test_legs_clip_to_the_range(self) -> None:
        sym = self._sym(_frame(left_hip=0.9 * math.pi, right_hip=0.0, left_knee=0.9 * math.pi, right_knee=0.0))
        self.assertAlmostEqual(sym[SymmetryElement.legs], 1.0, places=5)

    def test_a_missing_side_is_nan_with_zero_score(self) -> None:
        sym = self._sym(_frame(left_shoulder=0.5, left_elbow=0.5, right_elbow=0.5))
        self.assertTrue(math.isnan(sym[SymmetryElement.shoulder]))
        self.assertEqual(sym.get_score(SymmetryElement.shoulder), 0.0)
        self.assertTrue(math.isnan(sym[SymmetryElement.arms]))
        self.assertEqual(sym.get_score(SymmetryElement.arms), 0.0)
        self.assertAlmostEqual(sym[SymmetryElement.elbow], 0.0, places=5)
        self.assertEqual(sym.get_score(SymmetryElement.elbow), 1.0)

    def test_all_nan_leaves_feature_absent(self) -> None:
        out = AngleSymExtractor(_settings()).process(_frame())
        self.assertNotIn(AngleSymmetry, out)
        self.assertTrue(np.isnan(out[AngleSymmetry].values).all())
        np.testing.assert_array_equal(out[AngleSymmetry].scores, 0.0)

    def test_a_pair_score_is_the_lower_of_its_joints(self) -> None:
        f = _frame(left_shoulder=0.5, right_shoulder=0.5)
        scores = f[Angles].scores.copy()
        scores[AngleLandmark.right_shoulder] = 0.4
        f = Frame(track_id=0, cam_id=0, features={Angles: Angles(f[Angles].values.copy(), scores)})
        self.assertAlmostEqual(self._sym(f).get_score(SymmetryElement.shoulder), 0.4, places=5)


if __name__ == "__main__":
    unittest.main()
