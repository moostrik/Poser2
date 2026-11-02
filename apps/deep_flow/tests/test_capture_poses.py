"""CapturePoses: the center view's pose is the one in the camera's newest capture."""

import unittest
from typing import cast

from apps.deep_flow.capture_poses import CapturePoses
from modules.camera import CameraFrame
from modules.inference import Capture, GpuImage
from modules.pose import BBox, Frame
from modules.utils import Rect


def pose(cam_id: int, timestamp: float) -> Frame:
    return Frame(track_id=cam_id, cam_id=cam_id, timestamp=timestamp, features={BBox: BBox.from_rect(Rect(0.2, 0.2, 0.5, 0.5))})


def capture(cam_id: int, poses: dict[int, Frame]) -> Capture:
    return Capture(cam_id=cam_id, timestamp=1.0, image=cast(GpuImage, None), depth=cast(CameraFrame, None), poses=poses, poses_synced=True)


class FakeBoard:
    def __init__(self) -> None:
        self.captures: dict[int, Capture] = {}

    def get_capture(self, cam_id: int) -> Capture | None:
        return self.captures.get(cam_id)

    def set_capture(self, capture: Capture) -> None:
        self.captures[capture.cam_id] = capture


class CapturePosesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.board = FakeBoard()
        self.poses = CapturePoses(self.board)

    def test_the_pose_is_the_one_in_the_newest_capture_whatever_stage_is_asked(self) -> None:
        own = pose(0, 1.0)
        self.board.set_capture(capture(0, {0: own}))
        self.assertIs(self.poses.get_frame(0, 0), own)
        self.assertIs(self.poses.get_frame(3, 0), own)

    def test_no_capture_or_no_pose_is_none(self) -> None:
        self.assertIsNone(self.poses.get_frame(0, 0))
        self.board.set_capture(capture(0, {}))
        self.assertIsNone(self.poses.get_frame(0, 0))


if __name__ == '__main__':
    unittest.main()
