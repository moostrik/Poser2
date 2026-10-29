"""The tracker's intake: a camera's tracklet becomes the tracker's, with the camera's facts kept."""

import unittest

from modules.camera import CameraTracklet, TrackingStatus
from modules.tracker import Tracklet
from modules.utils import Rect


class FromCameraTest(unittest.TestCase):
    def test_the_camera_facts_are_kept(self) -> None:
        reported = CameraTracklet(cam_id=2, id=7, status=TrackingStatus.LOST, roi=Rect(0.1, 0.2, 0.3, 0.4), age=12, timestamp=1.5)
        tracklet = Tracklet.from_camera(reported)
        self.assertEqual(tracklet.cam_id, 2)
        self.assertEqual(tracklet.external_id, 7)
        self.assertEqual(tracklet.external_age_in_frames, 12)
        self.assertEqual(tracklet.status, TrackingStatus.LOST)
        self.assertEqual(tracklet.roi, Rect(0.1, 0.2, 0.3, 0.4))
        self.assertEqual(tracklet.id, -1)                        # the world identity is the tracker's to decide
        self.assertTrue(tracklet.is_lost)


if __name__ == '__main__':
    unittest.main()
