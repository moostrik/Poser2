"""The two-model chain's join: a tracklet gets the keypoints of the person whose box is its source
detection, whichever of the two streams arrives second. Runs on a depthai 3 install."""

import unittest

import numpy as np

from modules.camera import CameraTracklet, TrackingStatus
from modules.camera.oak4.pose_join import PoseJoin
from modules.utils import Rect


def tracklet(id: int) -> CameraTracklet:
    return CameraTracklet(cam_id=0, id=id, status=TrackingStatus.TRACKED, roi=Rect(0.0, 0.0, 0.1, 0.1),
                          age=1, timestamp=0.0)


class PoseJoinTest(unittest.TestCase):
    def test_the_tracklet_takes_the_keypoints_of_its_detection(self) -> None:
        join = PoseJoin()
        near = (0.1, 0.1, 0.3, 0.9)
        far = (0.6, 0.4, 0.7, 0.7)
        points_near = np.full((17, 3), 0.25, dtype=np.float32)
        points_far = np.full((17, 3), 0.75, dtype=np.float32)
        # The tracker lists the far person first; the detections and their poses have the near one first.
        self.assertIsNone(join.put_tracklets(7, [tracklet(2), tracklet(1), tracklet(9)], [far, near, (0.0, 0.0, 0.0, 0.0)]))
        joined = join.put_poses(7, [near, far], [points_near, points_far])
        self.assertEqual([t.id for t in joined], [2, 1, 9])
        np.testing.assert_array_equal(joined[0].keypoints, points_far)
        np.testing.assert_array_equal(joined[1].keypoints, points_near)
        self.assertIsNone(joined[2].keypoints)          # a tracklet without a detection this frame
        self.assertEqual(join.frames, {})

    def test_a_frame_that_never_completes_is_dropped(self) -> None:
        join = PoseJoin(keep=2)
        join.put_poses(1, [], [])
        join.put_poses(4, [], [])
        self.assertEqual(join.dropped, 1)
        self.assertEqual(set(join.frames), {4})


if __name__ == '__main__':
    unittest.main()
