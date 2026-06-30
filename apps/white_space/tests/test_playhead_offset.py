"""Tests for the radian PlayheadOffset / SingleAngle / azimuth-vs-playhead crossing."""

import math
import unittest

from modules.pose.frame import Frame
from modules.pose.features import Azimuth
from apps.white_space.pose import PlayheadOffset, PlayheadOffsetExtractor
from apps.white_space.light.layers._utilities import angle_to_strip_position
from apps.white_space.light.layers.low.playhead_flash import offset_to_level

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


class PlayheadOffsetExtractorTest(unittest.TestCase):
    def _offset(self, azimuth: float, playhead: float) -> float:
        return PlayheadOffsetExtractor(lambda: playhead).process(_frame(azimuth))[PlayheadOffset].value

    def test_on_playhead_is_zero(self) -> None:
        self.assertAlmostEqual(self._offset(0.3, 0.3), 0.0, places=4)

    def test_quarter_ahead(self) -> None:
        self.assertAlmostEqual(self._offset(0.3 + PI / 2, 0.3), PI / 2, places=4)

    def test_quarter_behind(self) -> None:
        self.assertAlmostEqual(self._offset(0.3 - PI / 2, 0.3), -PI / 2, places=4)

    def test_opposite_side_is_pi(self) -> None:
        self.assertAlmostEqual(abs(self._offset(0.0, PI)), PI, places=4)

    def test_wraps_shortest(self) -> None:
        # azimuth just past +π, playhead just before -π → small positive shortest diff
        self.assertAlmostEqual(self._offset(-PI + 0.1, PI - 0.1), 0.2, places=4)

    def test_missing_inputs_absent(self) -> None:
        self.assertNotIn(PlayheadOffset, PlayheadOffsetExtractor(lambda: 0.5).process(_frame(float("nan"))))
        self.assertNotIn(PlayheadOffset, PlayheadOffsetExtractor(lambda: float("nan")).process(_frame(0.5)))


class PlayheadFlashWindowTest(unittest.TestCase):
    def test_on_within_window(self) -> None:
        self.assertEqual(offset_to_level(0.0, 0.5, 0.5), 1.0)     # on the playhead
        self.assertEqual(offset_to_level(0.25, 0.5, 0.5), 1.0)    # ahead, within rise
        self.assertEqual(offset_to_level(-0.25, 0.5, 0.5), 1.0)   # behind, within fall
        self.assertEqual(offset_to_level(0.5, 0.5, 0.5), 1.0)     # edge of rise (inclusive)
        self.assertEqual(offset_to_level(-0.5, 0.5, 0.5), 1.0)    # edge of fall (inclusive)

    def test_off_outside_and_nan(self) -> None:
        self.assertEqual(offset_to_level(1.0, 0.5, 0.5), 0.0)        # ahead, beyond rise
        self.assertEqual(offset_to_level(-1.0, 0.5, 0.5), 0.0)       # behind, beyond fall
        self.assertEqual(offset_to_level(float("nan"), 0.5, 0.5), 0.0)
        self.assertEqual(offset_to_level(0.3, 0.0, 0.0), 0.0)       # zero widths → off

    def test_asymmetric_window(self) -> None:
        # wide rise, narrow fall
        self.assertEqual(offset_to_level(0.9, 1.0, 0.2), 1.0)    # ahead, within wide rise
        self.assertEqual(offset_to_level(-0.1, 1.0, 0.2), 1.0)   # behind, within narrow fall
        self.assertEqual(offset_to_level(-0.3, 1.0, 0.2), 0.0)   # behind, beyond narrow fall


if __name__ == "__main__":
    unittest.main()
