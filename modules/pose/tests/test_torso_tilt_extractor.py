"""Tests for TorsoTiltExtractor — signed sideways lean of the spine against the image
vertical, in isotropic (aspect-corrected) space."""

import math
import unittest

import numpy as np

from modules.pose.frame import Frame
from modules.pose.features import Points2D, PointLandmark, TorsoTilt
from modules.pose.nodes import TorsoTiltExtractor, TorsoTiltExtractorSettings

ASPECT = 0.75
TILT_DEGREES = 45.0                       # the setting
TILT_RAD = math.radians(TILT_DEGREES)     # the same lean as frame geometry


def _frame(hip_mid: tuple[float, float], shoulder_mid: tuple[float, float],
           drop: PointLandmark | None = None, score: float = 1.0) -> Frame:
    """A pose frame with hips and shoulders symmetric about the given midpoints (image
    coordinates, y downward); all other keypoints NaN."""
    n = len(PointLandmark)
    values = np.full((n, 2), np.nan, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    half = 0.1
    for lm, (mx, my), side in ((PointLandmark.left_hip, hip_mid, -1), (PointLandmark.right_hip, hip_mid, 1),
                               (PointLandmark.left_shoulder, shoulder_mid, -1), (PointLandmark.right_shoulder, shoulder_mid, 1)):
        values[lm] = (mx + side * half, my)
        scores[lm] = score
    if drop is not None:
        values[drop] = (np.nan, np.nan)
        scores[drop] = 0.0
    return Frame(track_id=0, cam_id=0, features={Points2D: Points2D(values, scores)})


def _lean(angle: float, spine: float = 0.4) -> Frame:
    """Hips at the crop centre, shoulders ``spine`` (isotropic units) up the spine leaning
    ``angle`` radians toward image right — built in isotropic space, then squashed back
    to crop coordinates (y × aspect) so the extractor has to undo it."""
    hx, hy = 0.5, 0.7
    dx = spine * math.sin(angle)
    dy = -spine * math.cos(angle)          # up = negative y
    return _frame((hx, hy), (hx + dx, hy + dy * ASPECT))


class TorsoTiltExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        cfg = TorsoTiltExtractorSettings()
        cfg.tilt_degrees = TILT_DEGREES
        cfg.aspect_ratio = ASPECT
        self.extractor = TorsoTiltExtractor(cfg)

    def _tilt(self, frame: Frame) -> float:
        return self.extractor.process(frame)[TorsoTilt].value

    def test_upright_is_zero(self) -> None:
        self.assertAlmostEqual(self._tilt(_lean(0.0)), 0.0, places=5)

    def test_lean_right_is_positive_one_at_full_lean(self) -> None:
        self.assertAlmostEqual(self._tilt(_lean(TILT_RAD)), 1.0, places=4)

    def test_lean_left_is_negative(self) -> None:
        self.assertAlmostEqual(self._tilt(_lean(-TILT_RAD / 2.0)), -0.5, places=4)

    def test_beyond_full_lean_clamps(self) -> None:
        self.assertAlmostEqual(self._tilt(_lean(math.pi / 2.0 - 0.1)), 1.0, places=5)

    def test_aspect_correction_matters(self) -> None:
        # Without the y correction the same crop-space geometry reads as a bigger lean.
        cfg = TorsoTiltExtractorSettings()
        cfg.tilt_degrees = TILT_DEGREES
        cfg.aspect_ratio = 1.0
        uncorrected = TorsoTiltExtractor(cfg).process(_lean(TILT_RAD / 2.0))[TorsoTilt].value
        self.assertGreater(uncorrected, self._tilt(_lean(TILT_RAD / 2.0)))

    def test_missing_keypoint_leaves_feature_absent(self) -> None:
        out = self.extractor.process(_frame((0.5, 0.7), (0.5, 0.3), drop=PointLandmark.left_hip))
        self.assertNotIn(TorsoTilt, out)
        self.assertTrue(math.isnan(out[TorsoTilt].value))
        self.assertEqual(out[TorsoTilt].score, 0.0)

    def test_degenerate_spine_leaves_feature_absent(self) -> None:
        out = self.extractor.process(_frame((0.5, 0.7), (0.5, 0.7)))
        self.assertNotIn(TorsoTilt, out)

    def test_score_is_min_of_spine_keypoints(self) -> None:
        out = self.extractor.process(_frame((0.5, 0.7), (0.5, 0.3), score=0.4))
        self.assertAlmostEqual(out[TorsoTilt].score, 0.4, places=5)


if __name__ == "__main__":
    unittest.main()
