"""Tests for the replay camera: the clip frames become stamped contract frames, the detector's
boxes become tracklets through the tracker, with their frame's time, and the readings follow."""

import time
import unittest

import numpy as np

from modules.camera import CameraTracklet, FrameMetadata, FrameType, RigCameraSettings, TrackingStatus
from modules.camera.replay import Camera, Track
from modules.utils import Rect


class FakePlayer:
    def __init__(self) -> None:
        self.callbacks = []
        self.jump_callbacks = []
        self.nv12 = False

    def add_frame_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def add_jump_callback(self, callback) -> None:
        self.jump_callbacks.append(callback)

    def jump(self) -> None:
        for c in self.jump_callbacks:
            c()

    def get_chunk_info(self, cam_id: int) -> None:
        return None

    def push(self, cam_id: int, image: np.ndarray, frame_type: FrameType = FrameType.VIDEO, captured: float | None = None,
             metadata: FrameMetadata | None = None) -> None:
        for c in self.callbacks:
            c(cam_id, frame_type, image, metadata, captured)


class FakePersonDetector:
    def __init__(self, enabled: bool = True) -> None:
        self.callbacks = []
        self.enabled = enabled

    def add_detection_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def report(self, cam_id: int, boxes: list[Rect], timestamp: float = 0.0) -> None:
        for c in self.callbacks:
            c(FakeDetection(cam_id, boxes, timestamp))


class FakeDetection:
    def __init__(self, cam_id: int, boxes: list[Rect], timestamp: float) -> None:
        self.cam_id = cam_id
        self.boxes = boxes
        self.timestamp = timestamp


class FakeTracker:
    """Every box is its own track, numbered in order."""
    def update(self, boxes: list[Rect]) -> list[Track]:
        return [Track(id=i, status=TrackingStatus.TRACKED, roi=box, age=1) for i, box in enumerate(boxes)]


class ReplayCameraTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = RigCameraSettings()
        self.settings.initialize()
        self.player = FakePlayer()
        self.detector = FakePersonDetector()
        self.camera = Camera(self.player, self.settings, self.detector, FakeTracker(), id=1)
        self.frames = []
        self.messages = []
        self.camera.add_frame_callback(self.frames.append)
        self.camera.add_tracklet_callback(self.messages.append)
        self.camera.start()

    def test_a_clip_frame_of_this_camera_is_stamped_and_handed_over(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        before = time.perf_counter()
        self.player.push(0, image)          # another camera's frame
        self.player.push(1, image)
        self.assertEqual(len(self.frames), 1)
        self.assertEqual(self.frames[0].cam_id, 1)
        self.assertIs(self.frames[0].image, image)
        self.assertGreaterEqual(self.frames[0].timestamp, before)
        self.assertTrue(self.camera.connected)

    def test_a_frame_is_stamped_with_its_capture_on_the_play_s_clock(self) -> None:
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8), captured=12.5)
        self.assertEqual(self.frames[0].timestamp, 12.5)
        assert self.frames[0].arrival is not None
        self.assertGreater(self.frames[0].arrival, 12.5)

    def test_a_recorded_frame_s_depth_takes_its_video_s_time(self) -> None:
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8))
        time.sleep(0.003)
        self.player.push(1, np.zeros((4, 6), dtype=np.uint16), FrameType.DEPTH)
        self.assertEqual([f.kind for f in self.frames], [FrameType.VIDEO, FrameType.DEPTH])
        self.assertEqual(self.frames[1].timestamp, self.frames[0].timestamp)

    def test_the_detector_boxes_become_tracklets_with_their_frame_s_time(self) -> None:
        self.detector.report(1, [Rect(0.1, 0.2, 0.3, 0.4), Rect(0.5, 0.5, 0.2, 0.3)], 12.5)
        self.detector.report(0, [Rect(0.1, 0.2, 0.3, 0.4)])         # not this camera
        self.assertEqual(len(self.messages), 1)
        message = self.messages[0]
        tracklets, t = message.tracklets, message.timestamp
        self.assertEqual(message.cam_id, 1)
        self.assertEqual([tr.id for tr in tracklets], [0, 1])
        self.assertEqual(tracklets[0].roi, Rect(0.1, 0.2, 0.3, 0.4))
        self.assertEqual(tracklets[0].cam_id, 1)
        self.assertEqual(t, 12.5)
        self.assertTrue(all(tr.timestamp == t for tr in tracklets))
        self.assertEqual(self.settings.readings.tracklets, 2)

    def test_nobody_is_a_message_too(self) -> None:
        self.detector.report(1, [])
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0].tracklets, [])

    def test_stopped_it_emits_nothing(self) -> None:
        self.camera.stop()
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8))
        self.detector.report(1, [Rect(0.1, 0.2, 0.3, 0.4)])
        self.assertEqual(self.frames, [])
        self.assertEqual(self.messages, [])
        self.assertFalse(self.camera.connected)


def recorded(timestamp: float, latency: float | None = 0.03, tracklets: bool = True) -> FrameMetadata:
    """A recorded frame's metadata with one tracklet of the camera's, `latency` after the capture."""
    found = [CameraTracklet(cam_id=0, id=5, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=9, timestamp=timestamp)]
    return FrameMetadata(timestamp, tracklets=found if tracklets else None, tracklet_latency=latency)


def wait_for(condition, seconds: float = 2.0) -> bool:
    end: float = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.002)
    return condition()


class RecordedTrackletsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = RigCameraSettings()
        self.settings.initialize()
        self.player = FakePlayer()
        self.detector = FakePersonDetector(enabled=False)
        self.camera = Camera(self.player, self.settings, self.detector, FakeTracker(), id=1)
        self.messages = []
        self.camera.add_tracklet_callback(lambda message: self.messages.append((time.perf_counter(), message)))
        self.camera.start()

    def tearDown(self) -> None:
        self.camera.stop()
        self.camera.join(1.0)

    def test_a_frame_s_tracklets_come_after_their_latency_with_its_stamp(self) -> None:
        captured = time.perf_counter()
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8), captured=captured, metadata=recorded(10.0, 0.03))
        self.assertTrue(wait_for(lambda: self.messages))
        came, message = self.messages[0]
        self.assertGreaterEqual(came, captured + 0.029)
        self.assertEqual((message.cam_id, message.timestamp, message.straightened), (1, captured, False))
        self.assertEqual((message.tracklets[0].cam_id, message.tracklets[0].id, message.tracklets[0].timestamp), (1, 5, captured))
        self.assertIsNotNone(message.arrival)

    def test_a_frame_none_came_for_sends_none(self) -> None:
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8), captured=time.perf_counter(), metadata=recorded(10.0, tracklets=False))
        time.sleep(0.05)
        self.assertEqual(self.messages, [])

    def test_a_jump_drops_what_waits(self) -> None:
        now = time.perf_counter()
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8), captured=now, metadata=recorded(10.0, 0.2))
        self.player.jump()                                                                                         # a loop
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8), captured=now + 0.01, metadata=recorded(2.0, 0.0))
        self.assertTrue(wait_for(lambda: self.messages))
        time.sleep(0.25)
        self.assertEqual([m.timestamp for _, m in self.messages], [now + 0.01])

    def test_with_the_detector_enabled_its_tracklets_play_instead(self) -> None:
        self.detector.enabled = True
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8), captured=time.perf_counter(), metadata=recorded(10.0, 0.0))
        time.sleep(0.05)
        self.assertEqual(self.messages, [])
        self.detector.report(1, [Rect(0.1, 0.2, 0.3, 0.4)], 12.5)
        self.assertTrue(self.messages[0][1].straightened)


class WithoutDetectorTest(unittest.TestCase):
    def test_frames_only(self) -> None:
        settings = RigCameraSettings()
        settings.initialize()
        player = FakePlayer()
        camera = Camera(player, settings, None, FakeTracker(), id=0)
        frames = []
        camera.add_frame_callback(frames.append)
        camera.start()
        player.push(0, np.zeros((4, 6), dtype=np.uint8))
        self.assertEqual(len(frames), 1)
        self.assertIsNone(camera.tracker)


if __name__ == '__main__':
    unittest.main()
