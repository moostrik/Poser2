"""The data layers' angle offset: each shift and which features it applies to. No GL."""

import math
import unittest

import numpy as np

from modules.pose.features import Angles, AngleMotion, AngleVelocity, Azimuth
from modules.render.layers.data.DataLayerSettings import AngleOffset, has_pi_range, offset_angles

# hanging (0), horizontal out (π/2), raised (π) and across the body (-π/2)
POSES = np.array([0.0, math.pi / 2, math.pi, -math.pi / 2], dtype=np.float32)


class TestOffsetAngles(unittest.TestCase):

    def test_none_returns_the_input(self) -> None:
        self.assertIs(offset_angles(POSES, AngleOffset.NONE), POSES)

    def test_half_pi_puts_the_seam_across_the_body(self) -> None:
        out = offset_angles(POSES, AngleOffset.HALF_PI)
        np.testing.assert_allclose(out, [-math.pi / 2, 0.0, math.pi / 2, -math.pi], atol=1e-6)

    def test_pi_puts_the_seam_at_hanging(self) -> None:
        out = offset_angles(POSES, AngleOffset.PI)
        np.testing.assert_allclose(out, [-math.pi, -math.pi / 2, 0.0, math.pi / 2], atol=1e-6)

    def test_half_pi_keeps_the_arc_from_hanging_to_raised_unbroken(self) -> None:
        arc = np.linspace(0.0, math.pi, 50, dtype=np.float32)
        self.assertTrue(np.all(np.diff(offset_angles(arc, AngleOffset.HALF_PI)) > 0.0))

    def test_pi_keeps_the_arc_around_raised_unbroken(self) -> None:
        arc = np.linspace(math.pi / 2, math.pi, 25, dtype=np.float32)
        out = offset_angles(np.concatenate([arc, -arc[::-1]]), AngleOffset.PI)   # π/2 → π ≡ -π → -π/2
        self.assertTrue(np.all(np.diff(out) > 0.0))

    def test_nan_passes_through(self) -> None:
        out = offset_angles(np.array([np.nan, 1.0], dtype=np.float32), AngleOffset.PI)
        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(float(out[1]), 1.0 - math.pi, places=6)

    def test_dtype_kept_and_input_untouched(self) -> None:
        values = np.array([[0.5, -0.5], [2.0, -2.0]], dtype=np.float32)
        before = values.copy()
        out = offset_angles(values, AngleOffset.HALF_PI)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, values.shape)
        np.testing.assert_array_equal(values, before)


class TestHasPiRange(unittest.TestCase):

    def test_wrapping_angle_features(self) -> None:
        self.assertTrue(has_pi_range(Angles))
        self.assertTrue(has_pi_range(Azimuth))

    def test_other_features(self) -> None:
        self.assertFalse(has_pi_range(AngleVelocity))   # unbounded range, only its display range is ±π
        self.assertFalse(has_pi_range(AngleMotion))     # normalized [0, 1]


if __name__ == '__main__':
    unittest.main()
