"""Tests for the radian PlayheadPhase / SingleAngle / azimuth-vs-playhead crossing."""

import math
import unittest

from modules.pose.frame import Frame
from modules.pose.features import Azimuth
from apps.white_space.pose import PlayheadPhase, PlayheadPhaseExtractor
from apps.white_space.light.sampler import sweep_contains
from apps.white_space.light.layers._utilities import angle_to_strip_position

PI = math.pi
TAU = math.tau


def _frame(azimuth: float) -> Frame:
    features = {} if math.isnan(azimuth) else {Azimuth: Azimuth.from_value(azimuth)}
    return Frame(track_id=0, cam_id=0, features=features)


class SingleAngleTest(unittest.TestCase):
    def test_from_value_wraps_to_pi(self) -> None:
        self.assertAlmostEqual(Azimuth.from_value(-PI / 2).value, -PI / 2, places=4)
        self.assertAlmostEqual(Azimuth.from_value(3 * PI / 2).value, -PI / 2, places=4)   # wrapped
        self.assertAlmostEqual(abs(Azimuth.from_value(PI).value), PI, places=4)            # boundary
        self.assertAlmostEqual(Azimuth.from_value(0.0).value, 0.0, places=4)

    def test_range_is_symmetric_pi(self) -> None:
        lo, hi = Azimuth.range()
        self.assertAlmostEqual(lo, -PI, places=6)
        self.assertAlmostEqual(hi, PI, places=6)

    def test_nan_preserved(self) -> None:
        dummy = Azimuth.from_value(float("nan"))
        self.assertTrue(math.isnan(dummy.value))


class AngleToStripPositionTest(unittest.TestCase):
    def test_round_trip_inverts_producer(self) -> None:
        # producer: world_angle (deg) -> Azimuth radians; layer: -> [0,1) == world_angle/360
        for deg in (0.0, 45.0, 179.0, 200.0, 359.0):
            az = Azimuth.from_value(math.radians(deg)).value
            self.assertAlmostEqual(angle_to_strip_position(az), (deg / 360.0) % 1.0, places=4)


class PlayheadPhaseExtractorTest(unittest.TestCase):
    def _phase(self, azimuth: float, playhead: float) -> float:
        return PlayheadPhaseExtractor(lambda: playhead).process(_frame(azimuth))[PlayheadPhase].value

    def test_on_playhead_is_zero(self) -> None:
        self.assertAlmostEqual(self._phase(0.3, 0.3), 0.0, places=4)

    def test_quarter_ahead(self) -> None:
        self.assertAlmostEqual(self._phase(0.3 + PI / 2, 0.3), PI / 2, places=4)

    def test_quarter_behind(self) -> None:
        self.assertAlmostEqual(self._phase(0.3 - PI / 2, 0.3), -PI / 2, places=4)

    def test_opposite_side_is_pi(self) -> None:
        self.assertAlmostEqual(abs(self._phase(0.0, PI)), PI, places=4)

    def test_wraps_shortest(self) -> None:
        # azimuth just past +π, playhead just before -π → small positive shortest diff
        self.assertAlmostEqual(self._phase(-PI + 0.1, PI - 0.1), 0.2, places=4)

    def test_missing_inputs_absent(self) -> None:
        self.assertNotIn(PlayheadPhase, PlayheadPhaseExtractor(lambda: 0.5).process(_frame(float("nan"))))
        self.assertNotIn(PlayheadPhase, PlayheadPhaseExtractor(lambda: float("nan")).process(_frame(0.5)))


class SweepContainsRadiansTest(unittest.TestCase):
    def test_flags_pose_when_swept(self) -> None:
        # playhead sweeps from 0.4 to 0.6 rad, pose at 0.5 → crossed
        self.assertTrue(sweep_contains(0.4, 0.6, 0.5))
        self.assertFalse(sweep_contains(0.4, 0.6, 0.7))

    def test_across_pi_wrap(self) -> None:
        # playhead wraps +π → -π; a pose near +π is crossed
        self.assertTrue(sweep_contains(PI - 0.05, -PI + 0.05, PI - 0.01))
        self.assertTrue(sweep_contains(PI - 0.05, -PI + 0.05, -PI + 0.01))
        self.assertFalse(sweep_contains(PI - 0.05, -PI + 0.05, 0.0))


if __name__ == "__main__":
    unittest.main()
