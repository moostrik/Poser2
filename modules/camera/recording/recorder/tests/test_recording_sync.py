"""`RecordingSync` through the recorder: a capture's color and depth paired, and the wait for its
tracklets and the poses asked for, settled by the streams' order."""

import unittest

from modules.camera import CameraFrame, FrameType, RecorderSettings
from modules.camera.recording.recorder.recorder import RecState
from modules.camera.recording.recorder.recording_sync import KEPT
from modules.camera.recording.tests.recordings import LENS, video_frame, poses, tracklets, depth_frame, recorder


class PairingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = recorder([FrameType.VIDEO, FrameType.DEPTH])
        self.recorder.state = RecState.REC

    def tearDown(self) -> None:
        self.recorder.settings.unbind(RecorderSettings.start, self.recorder._on_start)

    def test_a_depth_frame_takes_the_color_frame_captured_with_it(self) -> None:
        for i in range(3):
            self.recorder.submit_frame(video_frame(i))         # color arrives first
        self.recorder.submit_frame(depth_frame(1000, 1.0005))   # within the tolerance of frame 1
        capture = self.recorder.frames[0].get_nowait()
        self.assertEqual(capture.video.timestamp, 1.0)
        assert capture.depth is not None
        self.assertEqual(int(capture.depth.image[0, 0]), 1000)
        self.assertTrue(self.recorder.frames[0].empty())

    def test_a_color_frame_whose_depth_never_comes_is_left_out_and_counted(self) -> None:
        for i in range(3):
            self.recorder.submit_frame(video_frame(i))
        self.recorder.submit_frame(depth_frame(1000, 0.0))      # the first capture
        self.recorder.submit_frame(depth_frame(1000, 2.0))      # frame 1's depth was dropped
        timestamps = [self.recorder.frames[0].get_nowait().video.timestamp for _ in range(2)]
        self.assertEqual(timestamps, [0.0, 2.0])
        self.assertEqual(self.recorder.sync.missed.total('depth'), 1)

    def test_a_color_frame_captured_after_the_depth_waits_for_its_own(self) -> None:
        self.recorder.submit_frame(video_frame(5))
        self.recorder.submit_frame(depth_frame(1000, 4.0))      # no color frame of its own
        self.assertTrue(self.recorder.frames[0].empty())
        self.recorder.submit_frame(depth_frame(2000, 5.0))
        capture = self.recorder.frames[0].get_nowait()
        assert capture.depth is not None
        self.assertEqual((capture.video.timestamp, int(capture.depth.image[0, 0])), (5.0, 2000))

    def test_nothing_is_recorded_before_the_first_depth(self) -> None:
        self.recorder.submit_frame(video_frame(0))
        self.assertTrue(self.recorder.frames[0].empty())

    def test_the_frames_are_kept_as_delivered(self) -> None:
        frame = video_frame(0)
        mirrored = CameraFrame(0, FrameType.VIDEO, frame.image, 0.0, flip_h=True, intrinsics=LENS)
        self.recorder.submit_frame(mirrored)
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertIs(self.recorder.frames[0].get_nowait().video.image, frame.image)


class TrackletTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorder = recorder([FrameType.VIDEO, FrameType.DEPTH])
        self.recorder.state = RecState.REC

    def tearDown(self) -> None:
        self.recorder.settings.unbind(RecorderSettings.start, self.recorder._on_start)

    def test_a_capture_waits_for_its_tracklets(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))          # this camera sends tracklets
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))      # its tracklets are on their way
        self.assertEqual(self.recorder.frames[0].qsize(), 1)
        self.recorder.submit_tracklets(tracklets(1.0))
        captures = [self.recorder.frames[0].get_nowait() for _ in range(2)]
        self.assertEqual([c.tracklets.timestamp if c.tracklets else None for c in captures], [0.0, 1.0])

    def test_a_newer_capture_does_not_end_the_wait_and_the_order_holds(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        for i in (1, 2):
            self.recorder.submit_frame(video_frame(i))
            self.recorder.submit_frame(depth_frame(1000, float(i)))
        self.assertEqual(self.recorder.frames[0].qsize(), 1)    # frames 1 and 2 wait, however long
        self.recorder.submit_tracklets(tracklets(1.0))
        self.recorder.submit_tracklets(tracklets(2.0))
        self.assertEqual([self.recorder.frames[0].get_nowait().video.timestamp for _ in range(3)], [0.0, 1.0, 2.0])

    def test_a_later_frame_s_tracklets_settle_a_frame_without_its_own(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))
        self.recorder.submit_tracklets(tracklets(2.0))          # frame 1's never came
        captures = [self.recorder.frames[0].get_nowait() for _ in range(2)]
        self.assertIsNone(captures[1].tracklets)
        self.assertEqual(self.recorder.sync.missed.total('tracklets'), 1)

    def test_a_stream_that_stopped_holds_no_more_than_kept(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        for i in range(KEPT + 3):
            self.recorder.submit_frame(video_frame(i))
            self.recorder.submit_frame(depth_frame(1000, float(i)))
        self.assertEqual(self.recorder.sync.waiting, KEPT)

    def test_a_camera_without_tracklets_is_not_waited_for(self) -> None:
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertIsNone(self.recorder.frames[0].get_nowait().tracklets)
        self.assertEqual(self.recorder.sync.missed.total('tracklets'), 0)

    def test_a_capture_waits_for_the_poses_asked_for(self) -> None:
        self.recorder.expect_poses(0, 0.0, True)
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertTrue(self.recorder.frames[0].empty())
        self.recorder.submit_poses(poses(0.0))
        capture = self.recorder.frames[0].get_nowait()
        assert capture.poses is not None
        self.assertEqual(list(capture.poses.poses), [3])

    def test_a_frame_nobody_asked_poses_for_does_not_wait(self) -> None:
        self.recorder.expect_poses(0, 0.0, False)
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertIsNone(self.recorder.frames[0].get_nowait().poses)
        self.assertEqual(self.recorder.sync.missed.total('poses'), 0)

    def test_a_frame_not_yet_said_waits_to_know(self) -> None:
        self.recorder.expect_poses(0, 0.0, False)               # this app says per frame
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))      # not said yet for frame 1
        self.assertEqual(self.recorder.frames[0].qsize(), 1)
        self.recorder.expect_poses(0, 1.0, False)
        self.assertEqual(self.recorder.frames[0].qsize(), 2)

    def test_poses_asked_for_and_passed_by_a_later_frame_s_are_counted(self) -> None:
        self.recorder.expect_poses(0, 0.0, True)
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.assertTrue(self.recorder.frames[0].empty())
        self.recorder.submit_poses(poses(1.0))                  # the batch of frame 0 was dropped
        self.assertIsNone(self.recorder.frames[0].get_nowait().poses)
        self.assertEqual(self.recorder.sync.missed.total('poses'), 1)

    def test_a_later_frame_s_word_settles_a_frame_none_was_said_of(self) -> None:
        self.recorder.expect_poses(0, 0.0, False)
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))      # no word of frame 1: its image was dropped
        self.assertTrue(self.recorder.frames[0].empty())
        self.recorder.expect_poses(0, 2.0, False)
        self.assertIsNone(self.recorder.frames[0].get_nowait().poses)

    def test_a_waiting_capture_keeps_its_tracklets_past_kept_newer_ones(self) -> None:
        self.recorder.expect_poses(0, 0.0, True)
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))      # waits for its poses
        for i in range(1, KEPT + 2):
            self.recorder.submit_tracklets(tracklets(float(i)))  # its own tracklets pushed out of the stream
        self.recorder.submit_poses(poses(0.0))
        capture = self.recorder.frames[0].get_nowait()
        assert capture.tracklets is not None
        self.assertEqual(capture.tracklets.timestamp, 0.0)

    def test_the_end_writes_what_waits(self) -> None:
        self.recorder.submit_tracklets(tracklets(0.0))
        self.recorder.submit_frame(video_frame(0))
        self.recorder.submit_frame(depth_frame(1000, 0.0))
        self.recorder.submit_frame(video_frame(1))
        self.recorder.submit_frame(depth_frame(1000, 1.0))
        self.recorder.sync.flush()
        self.assertEqual(self.recorder.frames[0].qsize(), 2)


if __name__ == '__main__':
    unittest.main()
