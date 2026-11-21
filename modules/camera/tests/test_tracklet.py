"""`straightened_tracklets`: a camera's tracklets moved into the frame the uploader makes, straightened, then turned
and flipped."""

import unittest

import numpy as np

from modules.camera import CameraFrame, CameraTracklet, CameraTracklets, FrameType, TrackingStatus, straightened_tracklets, \
    oriented_image
from modules.camera.geometry import corrected_roi
from modules.utils import Rect

SHIFT: np.ndarray = np.array([[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def message(roi: Rect, keypoints: np.ndarray | None = None) -> CameraTracklets:
    tracklet = CameraTracklet(cam_id=0, id=3, status=TrackingStatus.TRACKED, roi=roi, age=5, timestamp=2.0, keypoints=keypoints)
    return CameraTracklets(cam_id=0, timestamp=2.0, tracklets=[tracklet], arrival=2.03)


class StraightenedTrackletsTest(unittest.TestCase):
    def test_the_box_is_straightened_as_corrected_roi_does(self) -> None:
        roi = Rect(0.1, 0.2, 0.3, 0.4)
        result = straightened_tracklets(message(roi), SHIFT, False, False, False)
        self.assertEqual(result.tracklets[0].roi, corrected_roi(roi, SHIFT))
        self.assertTrue(result.straightened)

    def test_straightened_first_then_mirrored(self) -> None:
        points = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
        result = straightened_tracklets(message(Rect(0.1, 0.2, 0.3, 0.4), points), SHIFT, False, False, True)
        roi = result.tracklets[0].roi
        self.assertAlmostEqual(roi.x, 1.0 - 0.35 - 0.3)
        keypoints = result.tracklets[0].keypoints
        assert keypoints is not None
        np.testing.assert_array_almost_equal(keypoints, [[1.0 - 0.35, 0.5, 0.9]])

    def test_turned_and_flipped_as_the_image_is(self) -> None:
        """A box and a keypoint land where `oriented_image` puts the pixels under them, for every flag set."""
        height, width = 20, 40
        roi = Rect(0.1, 0.2, 0.3, 0.4)
        point = np.array([[5.5 / 40, 12.5 / 20, 0.9]], dtype=np.float32)      # the center of pixel (12, 5)
        image = np.zeros((height, width), dtype=np.uint8)
        image[int(roi.y * height):int((roi.y + roi.height) * height), int(roi.x * width):int((roi.x + roi.width) * width)] = 1
        image[int(point[0, 1] * height), int(point[0, 0] * width)] = 2
        for rotate90 in (False, True):
            for flip_v in (False, True):
                for flip_h in (False, True):
                    flags = (rotate90, flip_v, flip_h)
                    upright = oriented_image(CameraFrame(0, FrameType.VIDEO, image, 0.0, rotate90=rotate90, flip_v=flip_v, flip_h=flip_h))
                    h, w = upright.shape
                    result = straightened_tracklets(message(roi, point), None, *flags).tracklets[0]
                    box = result.roi
                    rows, columns = np.nonzero(upright == 1)
                    self.assertAlmostEqual(box.x, columns.min() / w, msg=flags)
                    self.assertAlmostEqual(box.y, rows.min() / h, msg=flags)
                    self.assertAlmostEqual(box.width, (columns.max() + 1 - columns.min()) / w, msg=flags)
                    self.assertAlmostEqual(box.height, (rows.max() + 1 - rows.min()) / h, msg=flags)
                    assert result.keypoints is not None
                    x, y = result.keypoints[0, :2]
                    # the keypoint's pixel center, (column + 0.5) / width: within the pixel it marks
                    self.assertEqual(upright[min(int(y * h), h - 1), min(int(x * w), w - 1)], 2, msg=flags)

    def test_the_rest_of_the_message_stays(self) -> None:
        source = message(Rect(0.1, 0.2, 0.3, 0.4))
        result = straightened_tracklets(source, None, False, False, False)
        self.assertEqual((result.cam_id, result.timestamp, result.arrival), (0, 2.0, 2.03))
        self.assertEqual((result.tracklets[0].id, result.tracklets[0].age), (3, 5))
        self.assertEqual(result.tracklets[0].roi, source.tracklets[0].roi)
        self.assertFalse(source.straightened)


if __name__ == '__main__':
    unittest.main()
