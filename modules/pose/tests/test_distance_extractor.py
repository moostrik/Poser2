"""Tests for DistanceExtractor — maps bbox bottom (feet) to a normalized distance."""

import math
import unittest

from modules.utils import Rect
from modules.pose.frame import Frame
from modules.pose.features import BBox, Distance
from modules.pose.nodes import DistanceExtractor, DistanceExtractorSettings


def _frame_with_bottom(bottom: float, height: float = 0.4) -> Frame:
    """Build a pose frame whose bbox bottom edge sits at ``bottom``.

    ``BBox.from_rect`` round-trips so that ``to_rect().bottom == y + height``.
    """
    rect = Rect(x=0.0, y=bottom - height, width=0.2, height=height)
    return Frame(track_id=0, cam_id=0, features={BBox: BBox.from_rect(rect)})


class DistanceExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        # near_y=0.9 (closest → 0.0), far_y=0.3 (farthest → 1.0), span=0.6
        cfg = DistanceExtractorSettings()
        cfg.near_y = 0.9
        cfg.far_y = 0.3
        self.extractor = DistanceExtractor(cfg)

    def _distance(self, bottom: float) -> float:
        out = self.extractor.process(_frame_with_bottom(bottom))
        return out[Distance].value

    def test_near_threshold_is_zero(self) -> None:
        self.assertAlmostEqual(self._distance(0.9), 0.0, places=5)

    def test_far_threshold_is_one(self) -> None:
        self.assertAlmostEqual(self._distance(0.3), 1.0, places=5)

    def test_midpoint_is_half(self) -> None:
        self.assertAlmostEqual(self._distance(0.6), 0.5, places=5)

    def test_closer_than_near_clamps_to_zero(self) -> None:
        self.assertAlmostEqual(self._distance(1.0), 0.0, places=5)

    def test_farther_than_far_clamps_to_one(self) -> None:
        self.assertAlmostEqual(self._distance(0.1), 1.0, places=5)

    def test_missing_bbox_leaves_distance_absent(self) -> None:
        out = self.extractor.process(_frame_with_bottom(float("nan"), height=0.0))
        self.assertNotIn(Distance, out)
        self.assertTrue(math.isnan(out[Distance].value))
        self.assertEqual(out[Distance].score, 0.0)

    def test_zero_span_yields_zero(self) -> None:
        cfg = DistanceExtractorSettings()
        cfg.near_y = 0.5
        cfg.far_y = 0.5
        extractor = DistanceExtractor(cfg)
        out = extractor.process(_frame_with_bottom(0.5))
        self.assertAlmostEqual(out[Distance].value, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
