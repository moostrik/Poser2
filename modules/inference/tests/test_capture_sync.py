"""CaptureSync: the image, the depth and the poses of one moment of capture, matched by time; without a GPU or a camera."""

import time
import unittest
from typing import cast

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import Capture, CaptureSync, CaptureSyncSettings, GpuImage
from modules.pose import BBox, Frame, FrameDict
from modules.utils import Rect

FRAME: float = 1.0 / 60.0


def image(t: float, cam_id: int = 0) -> GpuImage:
    return GpuImage(cam_id, t, torch.zeros(1), cast(torch.cuda.Event, None))


def depth(t: float, cam_id: int = 0) -> CameraFrame:
    return CameraFrame(cam_id, FrameType.DEPTH, np.zeros((2, 2), np.uint16), t)


def poses(t: float, cam_id: int = 0) -> FrameDict:
    return {0: Frame(track_id=0, cam_id=cam_id, timestamp=t, features={BBox: BBox.from_rect(Rect(0.2, 0.2, 0.5, 0.5))})}


def wait_for(condition, timeout: float = 1.0) -> bool:
    end: float = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.001)
    return condition()


class CaptureSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = CaptureSyncSettings()
        self.settings.initialize()
        self.sync = CaptureSync(self.settings)
        self.captures: list[Capture] = []
        self.sync.add_capture_callback(self.captures.append)
        self.sync.start()

    def tearDown(self) -> None:
        self.sync.stop()

    def test_an_image_and_its_depth_make_a_capture_when_nobody_is_in_view(self) -> None:
        self.sync.submit_image(image(1.0))
        self.assertEqual(self.captures, [])
        self.sync.submit_depth(depth(1.0005))                   # within the tolerance
        self.assertEqual(len(self.captures), 1)
        self.assertEqual((self.captures[0].timestamp, self.captures[0].poses, self.captures[0].poses_synced), (1.0005, {}, False))

    def test_a_video_frame_is_not_a_depth_frame(self) -> None:
        self.sync.submit_image(image(1.0))
        self.sync.submit_depth(CameraFrame(0, FrameType.VIDEO, np.zeros((2, 2), np.uint8), 1.0))
        self.assertEqual(self.captures, [])

    def test_poses_in_before_the_depth_are_synced(self) -> None:
        self.sync.submit_image(image(1.0))
        self.sync.submit_poses(poses(1.0))
        self.sync.submit_depth(depth(1.0))
        self.assertTrue(self.captures[0].poses_synced)
        self.assertEqual(self.captures[0].poses[0].timestamp, 1.0)

    def test_with_a_player_in_view_the_capture_waits_for_its_own_poses(self) -> None:
        self.sync.submit_poses(poses(1.0 - FRAME))              # a player is in view
        self.sync.submit_image(image(1.0))
        self.sync.submit_depth(depth(1.0))
        self.assertEqual(self.captures, [])                     # its poses are still on their way
        self.sync.submit_poses(poses(1.0))
        self.assertEqual(len(self.captures), 1)
        self.assertTrue(self.captures[0].poses_synced)

    def test_poses_that_do_not_come_in_time_leave_the_capture_with_the_ones_before(self) -> None:
        self.settings.pose_wait = 5.0
        self.sync.submit_poses(poses(1.0 - FRAME))
        self.sync.submit_image(image(1.0))
        start: float = time.perf_counter()
        self.sync.submit_depth(depth(1.0))
        self.assertTrue(wait_for(lambda: len(self.captures) == 1))
        self.assertGreaterEqual(time.perf_counter() - start, 0.004)
        self.assertFalse(self.captures[0].poses_synced)
        self.assertEqual(self.captures[0].poses[0].timestamp, 1.0 - FRAME)

    def test_after_a_wait_that_ran_out_the_next_capture_does_not_wait(self) -> None:
        self.settings.pose_wait = 5.0
        self.sync.submit_poses(poses(1.0 - FRAME))
        self.sync.submit_image(image(1.0))
        self.sync.submit_depth(depth(1.0))
        self.assertTrue(wait_for(lambda: len(self.captures) == 1))
        self.sync.submit_image(image(1.0 + FRAME))
        self.sync.submit_depth(depth(1.0 + FRAME))              # the player left: no poses waited for
        self.assertEqual(len(self.captures), 2)

    def test_a_depth_frame_that_never_met_its_image_is_a_miss(self) -> None:
        self.sync.submit_depth(depth(1.0))                      # its image was dropped
        self.sync.submit_image(image(1.0 + FRAME))
        self.sync.submit_depth(depth(1.0 + FRAME))
        self.assertEqual([c.timestamp for c in self.captures], [1.0 + FRAME])
        self.assertEqual(self.settings.misses, 1)

    def test_cameras_are_matched_apart(self) -> None:
        self.sync.submit_image(image(1.0, cam_id=0))
        self.sync.submit_depth(depth(1.0, cam_id=1))
        self.assertEqual(self.captures, [])
        self.sync.submit_image(image(1.0, cam_id=1))
        self.assertEqual([(c.cam_id, c.timestamp) for c in self.captures], [(1, 1.0)])


if __name__ == '__main__':
    unittest.main()
