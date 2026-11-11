"""CaptureSync: the image, the depth, the poses and the masks of one moment of capture, matched by time; without a GPU or a camera."""

import time
import unittest
from typing import cast

import numpy as np
import torch

from modules.camera import CameraFrame, FrameType
from modules.inference import Capture, CaptureSync, CaptureSyncSettings, GpuImage, segmentation
from modules.pose import BBox, Frame, FrameDict
from modules.utils import Rect

FRAME: float = 1.0 / 60.0


def image(t: float, cam_id: int = 0) -> GpuImage:
    return GpuImage(cam_id, t, torch.zeros(1), cast(torch.cuda.Event, None))


def depth(t: float, cam_id: int = 0) -> CameraFrame:
    return CameraFrame(cam_id, FrameType.DEPTH, np.zeros((2, 2), np.uint16), t)


def poses(t: float, cam_id: int = 0) -> FrameDict:
    return {0: Frame(track_id=0, cam_id=cam_id, timestamp=t, features={BBox: BBox.from_rect(Rect(0.2, 0.2, 0.5, 0.5))})}


def masks(t: float, cam_id: int = 0) -> tuple[FrameDict, segmentation.ImageDict]:
    """RVM's result for the frame captured at `t`: the poses it was cut for, and their masks."""
    return poses(t, cam_id), {0: segmentation.Image(mask=torch.full((1,), t), foreground=torch.zeros(1))}


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

    def test_a_capture_waiting_for_its_poses_goes_out_when_a_newer_one_is_complete(self) -> None:
        self.sync.submit_poses(poses(1.0 - FRAME))              # a player is in view
        self.sync.submit_image(image(1.0))
        self.sync.submit_depth(depth(1.0))                      # waits for its poses
        self.sync.submit_image(image(1.0 + FRAME))
        self.sync.submit_depth(depth(1.0 + FRAME))              # complete: the waiting one goes out, with what it has,
        self.assertEqual([c.timestamp for c in self.captures], [1.0, 1.0 + FRAME])     # then this one, the poses behind
        self.assertFalse(self.captures[0].poses_synced or self.captures[1].poses_synced)
        self.assertEqual(self.settings.misses, 0)

    def test_complete_captures_go_out_in_the_order_of_capture(self) -> None:
        for i in range(5):
            self.sync.submit_image(image(1.0 + i * FRAME))
        for i in range(5):
            self.sync.submit_depth(depth(1.0 + i * FRAME))
        self.assertEqual([c.timestamp for c in self.captures], [1.0 + i * FRAME for i in range(5)])

    def test_with_pose_wait_zero_nothing_waits(self) -> None:
        self.settings.pose_wait = 0.0
        self.sync.submit_poses(poses(1.0 - FRAME))              # a player is in view
        self.sync.submit_image(image(1.0))
        self.sync.submit_depth(depth(1.0))
        self.assertEqual(len(self.captures), 1)
        self.assertEqual(self.captures[0].poses[0].timestamp, 1.0 - FRAME)

    def test_masks_in_before_the_depth_are_synced(self) -> None:
        self.sync.submit_image(image(1.0))
        self.sync.submit_poses(poses(1.0))
        self.sync.submit_masks(*masks(1.0))
        self.sync.submit_depth(depth(1.0))
        self.assertEqual(len(self.captures), 1)
        self.assertTrue(self.captures[0].masks_synced)
        self.assertEqual(float(self.captures[0].masks[0].mask[0]), 1.0)

    def test_while_masks_come_in_the_capture_waits_for_its_own(self) -> None:
        self.sync.submit_masks(*masks(1.0 - FRAME))             # RVM is running
        self.sync.submit_image(image(1.0))
        self.sync.submit_poses(poses(1.0))
        self.sync.submit_depth(depth(1.0))
        self.assertEqual(self.captures, [])                     # its masks are still on their way
        self.sync.submit_masks(*masks(1.0))
        self.assertEqual(len(self.captures), 1)
        self.assertTrue(self.captures[0].poses_synced and self.captures[0].masks_synced)

    def test_masks_that_do_not_come_in_time_leave_the_capture_with_the_ones_before(self) -> None:
        self.settings.mask_wait = 5.0
        self.sync.submit_masks(*masks(1.0 - FRAME))
        self.sync.submit_image(image(1.0))
        self.sync.submit_poses(poses(1.0))
        start: float = time.perf_counter()
        self.sync.submit_depth(depth(1.0))
        self.assertTrue(wait_for(lambda: len(self.captures) == 1))
        self.assertGreaterEqual(time.perf_counter() - start, 0.004)
        self.assertTrue(self.captures[0].poses_synced)
        self.assertFalse(self.captures[0].masks_synced)
        self.assertAlmostEqual(float(self.captures[0].masks[0].mask[0]), 1.0 - FRAME, places=5)

    def test_without_masks_the_capture_does_not_wait_for_them(self) -> None:
        self.sync.submit_image(image(1.0))
        self.sync.submit_poses(poses(1.0))
        self.sync.submit_depth(depth(1.0))                      # no RVM: nothing to wait for
        self.assertEqual(len(self.captures), 1)
        self.assertEqual((self.captures[0].masks, self.captures[0].masks_synced), ({}, False))

    def test_a_result_without_masks_passes(self) -> None:
        self.sync.submit_masks(poses(1.0), {})                  # RVM found no mask: nothing to sync
        self.sync.submit_image(image(1.0))
        self.sync.submit_depth(depth(1.0))
        self.assertEqual(len(self.captures), 1)

    def test_cameras_are_matched_apart(self) -> None:
        self.sync.submit_image(image(1.0, cam_id=0))
        self.sync.submit_depth(depth(1.0, cam_id=1))
        self.assertEqual(self.captures, [])
        self.sync.submit_image(image(1.0, cam_id=1))
        self.assertEqual([(c.cam_id, c.timestamp) for c in self.captures], [(1, 1.0)])


if __name__ == '__main__':
    unittest.main()
