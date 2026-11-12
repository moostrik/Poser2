"""`straightened_tracklets`: a camera's tracklets moved into the frame the uploader makes, straightened and then mirrored."""

import unittest

import numpy as np

from modules.camera import CameraTracklet, CameraTracklets, TrackingStatus, corrected_roi, straightened_tracklets
from modules.utils import Rect

SHIFT: np.ndarray = np.array([[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def message(roi: Rect, keypoints: np.ndarray | None = None) -> CameraTracklets:
    tracklet = CameraTracklet(cam_id=0, id=3, status=TrackingStatus.TRACKED, roi=roi, age=5, timestamp=2.0, keypoints=keypoints)
    return CameraTracklets(cam_id=0, timestamp=2.0, tracklets=[tracklet], arrival=2.03)


class StraightenedTrackletsTest(unittest.TestCase):
    def test_the_box_is_straightened_as_corrected_roi_does(self) -> None:
        roi = Rect(0.1, 0.2, 0.3, 0.4)
        result = straightened_tracklets(message(roi), SHIFT, False)
        self.assertEqual(result.tracklets[0].roi, corrected_roi(roi, SHIFT))
        self.assertTrue(result.straightened)

    def test_straightened_first_then_mirrored(self) -> None:
        points = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
        result = straightened_tracklets(message(Rect(0.1, 0.2, 0.3, 0.4), points), SHIFT, True)
        roi = result.tracklets[0].roi
        self.assertAlmostEqual(roi.x, 1.0 - 0.35 - 0.3)
        keypoints = result.tracklets[0].keypoints
        assert keypoints is not None
        np.testing.assert_array_almost_equal(keypoints, [[1.0 - 0.35, 0.5, 0.9]])

    def test_the_rest_of_the_message_stays(self) -> None:
        source = message(Rect(0.1, 0.2, 0.3, 0.4))
        result = straightened_tracklets(source, None, False)
        self.assertEqual((result.cam_id, result.timestamp, result.arrival), (0, 2.0, 2.03))
        self.assertEqual((result.tracklets[0].id, result.tracklets[0].age), (3, 5))
        self.assertEqual(result.tracklets[0].roi, source.tracklets[0].roi)
        self.assertFalse(source.straightened)


if __name__ == '__main__':
    unittest.main()
