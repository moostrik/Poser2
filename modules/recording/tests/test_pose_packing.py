"""The recorder's pose packing: the RAW stage's poses go in as they are and are packed into `FramePoses` on the sync's
thread, one per camera, the poses without points left out."""

import unittest

import numpy as np

from modules.pose import BBox, Frame, FrameDict, Points2D
from modules.utils import Rect

from .. import RecordingSync, frame_poses
from .recordings import BOX, POINTS, cam_frame, poses


def without_points(track_id: int, timestamp: float) -> Frame:
    return Frame(track_id=track_id, cam_id=0, timestamp=timestamp, features={BBox: BBox.from_rect(Rect(0.5, 0.5, 0.2, 0.2))})


class FramePosesTest(unittest.TestCase):
    def test_the_poses_with_points_per_camera(self) -> None:
        frames: FrameDict = poses(2.0) | {1: without_points(1, 2.0)}
        [kept] = frame_poses(frames, 2.05)
        self.assertEqual((kept.cam_id, kept.timestamp, kept.arrival, list(kept.poses)), (0, 2.0, 2.05, [3]))
        box, points = kept.poses[3]
        np.testing.assert_array_almost_equal(box, [BOX.x, BOX.y, BOX.width, BOX.height])
        np.testing.assert_array_almost_equal(points, POINTS)


class PosePackingTest(unittest.TestCase):
    def test_the_packed_poses_reach_the_capture(self) -> None:
        sync = RecordingSync(with_depth=False)
        sync.expect_parts(0, tracklets=False, poses=True)
        captures = []
        sync.add_capture_callback(captures.append)
        sync.start()
        try:
            sync.submit_frame(cam_frame(0, 0, 1.0))
            sync.expect_poses(0, 1.0, True)
            sync.submit_poses(poses(1.0), 1.05)
            sync.settle()
            self.assertEqual(len(captures), 1)
            self.assertIsNotNone(captures[0].poses)
            self.assertEqual(captures[0].poses.arrival, 1.05)
        finally:
            sync.stop()


if __name__ == '__main__':
    unittest.main()
