"""InferenceFrameSync: the image, the depth, the poses and the masks of one moment of capture, matched by time; without a GPU
or a camera."""

import threading
import time
import unittest
from typing import cast

import torch

from modules.inference import (Depth, GpuDepth, GpuImage, InferenceFrame, InferenceFrameSync, InferenceFrameSyncSettings,
                               InferenceResult, Masks, Poses, segmentation)
from modules.pose import BBox, Frame, FrameDict
from modules.utils import Rect

FRAME: float = 1.0 / 60.0


def image(t: float, cam_id: int = 0) -> GpuImage:
    return GpuImage(cam_id, t, torch.zeros(1), cast(torch.cuda.Event, None))


def depth(t: float, cam_id: int = 0) -> Depth:
    return Depth.of(GpuDepth(cam_id, t, torch.zeros(2, 2), cast(torch.cuda.Event, None)))


def players(t: float, cam_id: int = 0) -> FrameDict:
    return {0: Frame(track_id=0, cam_id=cam_id, timestamp=t, features={BBox: BBox.from_rect(Rect(0.2, 0.2, 0.5, 0.5))})}


def poses(t: float, cam_id: int = 0) -> Poses:
    return Poses.by_camera(players(t, cam_id))[0]


def masks(t: float, cam_id: int = 0) -> Masks:
    """The segmentation's result for the frame captured at `t`, found by the poses it was cut for."""
    return Masks.by_camera(players(t, cam_id), {0: segmentation.Image(mask=torch.full((1,), t), foreground=torch.zeros(1))})[0]


def wait_for(condition, timeout: float = 1.0) -> bool:
    end: float = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.001)
    return condition()


class InferenceFrameSyncTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = InferenceFrameSyncSettings()
        self.settings.initialize()
        self.sync = InferenceFrameSync(self.settings, (Depth, Poses, Masks))
        self.frames: list[InferenceFrame] = []
        self.came: list[float] = []
        self.threads: list[str] = []

        def on_frame(frame: InferenceFrame) -> None:
            self.came.append(time.perf_counter())
            self.frames.append(frame)
            self.threads.append(threading.current_thread().name)
        self.sync.add_frame_callback(on_frame)
        self.sync.start()

    def tearDown(self) -> None:
        self.sync.stop()

    def handed(self) -> list[InferenceFrame]:
        """The frames handed on once everything submitted is matched."""
        self.sync.settle()
        return self.frames

    def test_an_image_and_its_depth_make_a_frame_when_nobody_is_in_view(self) -> None:
        self.sync.submit_image(image(1.0))
        self.assertEqual(self.handed(), [])
        self.sync.submit(depth(1.0005))                   # within the tolerance
        self.assertEqual(len(self.handed()), 1)
        self.assertEqual((self.frames[0].timestamp, self.frames[0][Poses].poses, self.frames[0][Poses].synced), (1.0005, {}, False))

    def test_poses_in_before_the_depth_are_synced(self) -> None:
        self.sync.submit_image(image(1.0))
        self.sync.submit(poses(1.0))
        self.sync.submit(depth(1.0))
        self.assertTrue(self.handed()[0][Poses].synced)
        self.assertEqual(self.frames[0][Poses].poses[0].timestamp, 1.0)

    def test_with_a_player_in_view_the_frame_waits_for_its_own_poses(self) -> None:
        self.sync.submit(poses(1.0 - FRAME))              # a player is in view
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertEqual(self.handed(), [])                   # its poses are still on their way
        self.sync.submit(poses(1.0))
        self.assertEqual(len(self.handed()), 1)
        self.assertTrue(self.frames[0][Poses].synced)

    def test_a_frame_whose_poses_were_not_asked_goes_out_at_once_with_the_ones_before(self) -> None:
        self.sync.submit(poses(1.0 - FRAME))              # a player is in view, posed the frame before
        self.sync.expect(0, 1.0, False)                   # this frame was passed (`pose_interval`)
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertEqual(len(self.handed()), 1)               # no wait for poses that will not come
        self.assertEqual(self.frames[0][Poses].poses[0].timestamp, 1.0 - FRAME)

    def test_poses_asked_are_waited_for(self) -> None:
        self.sync.submit(poses(1.0 - FRAME))
        self.sync.expect(0, 1.0, True)
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertEqual(self.handed(), [])
        self.sync.submit(poses(1.0))
        self.assertTrue(self.handed()[0][Poses].synced)

    def test_poses_that_do_not_come_in_time_leave_the_frame_with_the_ones_before(self) -> None:
        self.settings.pose_wait = 5.0
        self.sync.submit(poses(1.0 - FRAME))
        self.sync.submit_image(image(1.0))
        start: float = time.perf_counter()
        self.sync.submit(depth(1.0))
        self.assertTrue(wait_for(lambda: len(self.frames) == 1))
        self.assertGreaterEqual(time.perf_counter() - start, 0.004)
        self.assertFalse(self.frames[0][Poses].synced)
        self.assertEqual(self.frames[0][Poses].poses[0].timestamp, 1.0 - FRAME)

    def test_a_frame_that_waits_goes_out_at_its_deadline_not_on_the_timer_tick(self) -> None:
        for wait_ms in (1.0, 10.0):
            self.settings.pose_wait = wait_ms
            t: float = 1.0 + len(self.frames) * 2 * FRAME
            self.sync.submit(poses(t - FRAME))
            self.sync.submit_image(image(t))
            self.sync.settle()
            start: float = time.perf_counter()
            self.sync.submit(depth(t))
            count: int = len(self.frames) + 1
            self.assertTrue(wait_for(lambda: len(self.frames) == count))
            waited: float = self.came[-1] - start
            self.assertGreaterEqual(waited, wait_ms / 1000.0)
            self.assertLess(waited, wait_ms / 1000.0 + 0.004, f'a {wait_ms:.0f} ms wait took {waited * 1000:.1f} ms')

    def test_after_a_wait_that_ran_out_the_next_frame_does_not_wait(self) -> None:
        self.settings.pose_wait = 5.0
        self.sync.submit(poses(1.0 - FRAME))
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertTrue(wait_for(lambda: len(self.frames) == 1))
        self.sync.submit_image(image(1.0 + FRAME))
        self.sync.submit(depth(1.0 + FRAME))              # the player left: no poses waited for
        self.assertEqual(len(self.handed()), 2)

    def test_a_depth_frame_that_never_met_its_image_is_a_miss(self) -> None:
        self.sync.submit(depth(1.0))                      # its image was dropped
        self.sync.submit_image(image(1.0 + FRAME))
        self.sync.submit(depth(1.0 + FRAME))
        self.assertEqual([c.timestamp for c in self.handed()], [1.0 + FRAME])
        self.assertEqual(self.settings.misses, 1)

    def test_a_frame_waiting_for_its_poses_goes_out_when_a_newer_one_is_complete(self) -> None:
        self.sync.submit(poses(1.0 - FRAME))              # a player is in view
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))                      # waits for its poses
        self.sync.submit_image(image(1.0 + FRAME))
        self.sync.submit(depth(1.0 + FRAME))              # complete: the waiting one goes out, with what it has,
        self.assertEqual([c.timestamp for c in self.handed()], [1.0, 1.0 + FRAME])  # then this one, the poses behind
        self.assertFalse(self.frames[0][Poses].synced or self.frames[1][Poses].synced)
        self.assertEqual(self.settings.misses, 0)

    def test_complete_frames_go_out_in_the_order_of_capture(self) -> None:
        for i in range(5):
            self.sync.submit_image(image(1.0 + i * FRAME))
        for i in range(5):
            self.sync.submit(depth(1.0 + i * FRAME))
        self.assertEqual([c.timestamp for c in self.handed()], [1.0 + i * FRAME for i in range(5)])

    def test_with_pose_wait_zero_nothing_waits(self) -> None:
        self.settings.pose_wait = 0.0
        self.sync.submit(poses(1.0 - FRAME))              # a player is in view
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertEqual(len(self.handed()), 1)
        self.assertEqual(self.frames[0][Poses].poses[0].timestamp, 1.0 - FRAME)

    def test_masks_in_before_the_depth_are_synced(self) -> None:
        self.sync.submit_image(image(1.0))
        self.sync.submit(poses(1.0))
        self.sync.submit(masks(1.0))
        self.sync.submit(depth(1.0))
        self.assertEqual(len(self.handed()), 1)
        self.assertTrue(self.frames[0][Masks].synced)
        self.assertEqual(float(self.frames[0][Masks].masks[0].mask[0]), 1.0)

    def test_while_masks_come_in_the_frame_waits_for_its_own(self) -> None:
        self.sync.submit(masks(1.0 - FRAME))             # RVM is running
        self.sync.submit_image(image(1.0))
        self.sync.submit(poses(1.0))
        self.sync.submit(depth(1.0))
        self.assertEqual(self.handed(), [])                   # its masks are still on their way
        self.sync.submit(masks(1.0))
        self.assertEqual(len(self.handed()), 1)
        self.assertTrue(self.frames[0][Poses].synced and self.frames[0][Masks].synced)

    def test_masks_that_do_not_come_in_time_leave_the_frame_with_the_ones_before(self) -> None:
        self.settings.mask_wait = 5.0
        self.sync.submit(masks(1.0 - FRAME))
        self.sync.submit_image(image(1.0))
        self.sync.submit(poses(1.0))
        start: float = time.perf_counter()
        self.sync.submit(depth(1.0))
        self.assertTrue(wait_for(lambda: len(self.frames) == 1))
        self.assertGreaterEqual(time.perf_counter() - start, 0.004)
        self.assertTrue(self.frames[0][Poses].synced)
        self.assertFalse(self.frames[0][Masks].synced)
        self.assertAlmostEqual(float(self.frames[0][Masks].masks[0].mask[0]), 1.0 - FRAME, places=5)

    def test_without_masks_the_frame_does_not_wait_for_them(self) -> None:
        self.sync.submit_image(image(1.0))
        self.sync.submit(poses(1.0))
        self.sync.submit(depth(1.0))                      # no RVM: nothing to wait for
        self.assertEqual(len(self.handed()), 1)
        self.assertEqual((self.frames[0][Masks].masks, self.frames[0][Masks].synced), ({}, False))

    def test_a_result_without_masks_passes(self) -> None:
        self.sync.submit(*Masks.by_camera(players(1.0), {}))    # the segmentation found no mask: nothing to sync
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertEqual(len(self.handed()), 1)

    def test_cameras_are_matched_apart(self) -> None:
        self.sync.submit_image(image(1.0, cam_id=0))
        self.sync.submit(depth(1.0, cam_id=1))
        self.assertEqual(self.handed(), [])
        self.sync.submit_image(image(1.0, cam_id=1))
        self.assertEqual([(c.cam_id, c.timestamp) for c in self.handed()], [(1, 1.0)])

    def test_frames_go_out_on_the_sync_thread(self) -> None:
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertEqual(len(self.handed()), 1)
        self.assertEqual(self.threads, ['InferenceFrameSync'])

    def test_a_submit_returns_while_the_sync_hands_on_a_frame(self) -> None:
        handed: threading.Event = threading.Event()
        release: threading.Event = threading.Event()

        def slow(frame: InferenceFrame) -> None:
            handed.set()
            release.wait(2.0)
        self.sync.add_frame_callback(slow)
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertTrue(handed.wait(1.0))
        start: float = time.perf_counter()
        self.sync.submit_image(image(1.0 + FRAME))              # the sync's thread is held in the callback
        self.assertLess(time.perf_counter() - start, 0.05)
        release.set()

    def test_a_stream_beyond_kept_drops_its_oldest_counted(self) -> None:
        handed: threading.Event = threading.Event()
        release: threading.Event = threading.Event()

        def hold(frame: InferenceFrame) -> None:
            handed.set()
            release.wait(2.0)
        self.sync.add_frame_callback(hold)
        self.sync.submit_image(image(1.0))
        self.sync.submit(depth(1.0))
        self.assertTrue(handed.wait(1.0))                       # the thread is held: what comes now waits in the inbox
        for i in range(1, 11):
            self.sync.submit_image(image(1.0 + i * FRAME))      # ten images, eight kept
        release.set()
        self.sync.settle()
        self.sync.submit_image(image(2.0))
        self.sync.submit(depth(2.0))                      # a frame, so the count is published
        self.sync.settle()
        self.assertEqual(self.settings.dropped, 2)


class InferenceFrameSyncLifecycleTest(unittest.TestCase):
    def test_a_part_before_start_is_an_error_and_one_after_stop_is_dropped(self) -> None:
        settings = InferenceFrameSyncSettings()
        settings.initialize()
        sync = InferenceFrameSync(settings, (Depth, Poses))
        frames: list[InferenceFrame] = []
        sync.add_frame_callback(frames.append)
        with self.assertRaises(RuntimeError):
            sync.submit_image(image(1.0))
        sync.start()
        sync.stop()
        sync.submit_image(image(1.0))
        sync.submit(depth(1.0))
        self.assertEqual(frames, [])


class InferenceFrameTypesTest(unittest.TestCase):
    """The types a sync is made with: only those are waited for; a frame reads the others as empty."""

    def make(self, results) -> tuple[InferenceFrameSync, list[InferenceFrame]]:
        settings = InferenceFrameSyncSettings()
        settings.initialize()
        sync = InferenceFrameSync(settings, results)
        frames: list[InferenceFrame] = []
        sync.add_frame_callback(frames.append)
        sync.start()
        self.addCleanup(sync.stop)
        return sync, frames

    def test_without_depth_the_image_alone_makes_a_frame(self) -> None:
        sync, frames = self.make((Poses,))
        sync.submit_image(image(1.0))
        sync.settle()
        self.assertEqual([f.timestamp for f in frames], [1.0])
        self.assertNotIn(Depth, frames[0])
        self.assertIsNone(frames[0][Depth].depth)

    def test_a_type_the_sync_was_not_made_with_is_never_waited_for_and_reads_empty(self) -> None:
        sync, frames = self.make((Depth, Poses))
        sync.submit(poses(1.0 - FRAME))
        sync.submit(poses(1.0))
        sync.submit_image(image(1.0))
        sync.submit(depth(1.0))
        sync.settle()
        self.assertEqual(len(frames), 1)
        self.assertEqual((frames[0][Masks].masks, frames[0][Masks].synced), ({}, False))

    def test_a_result_of_another_type_is_an_error(self) -> None:
        sync, _ = self.make((Depth, Poses))
        with self.assertRaises(ValueError):
            sync.submit(masks(1.0))

    def test_a_type_without_a_wait_cannot_be_given(self) -> None:
        settings = InferenceFrameSyncSettings()
        settings.initialize()
        with self.assertRaises(ValueError):
            InferenceFrameSync(settings, (InferenceResult,))


if __name__ == '__main__':
    unittest.main()
