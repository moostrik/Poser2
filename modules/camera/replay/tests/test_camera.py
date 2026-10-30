"""Tests for the replay camera: the clip frames become stamped contract frames, the detector's
boxes become tracklets through the tracker, with the message time, and the readings follow."""

import time
import unittest

import numpy as np

from modules.camera import FrameType, RigCameraSettings, TrackingStatus
from modules.camera.replay import Camera, Track
from modules.utils import Rect


class FakePlayer:
    def __init__(self) -> None:
        self.callbacks = []

    def add_frame_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def get_chunk_info(self, cam_id: int) -> None:
        return None

    def push(self, cam_id: int, image: np.ndarray, frame_type: FrameType = FrameType.VIDEO) -> None:
        for c in self.callbacks:
            c(cam_id, frame_type, image, None)


class FakePersonDetector:
    def __init__(self) -> None:
        self.submitted: list[tuple[int, np.ndarray]] = []
        self.callbacks = []

    def submit(self, cam_id: int, image: np.ndarray) -> None:
        self.submitted.append((cam_id, image))

    def add_detection_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def report(self, cam_id: int, boxes: list[Rect]) -> None:
        for c in self.callbacks:
            c(FakeDetection(cam_id, boxes))


class FakeDetection:
    def __init__(self, cam_id: int, boxes: list[Rect]) -> None:
        self.cam_id = cam_id
        self.boxes = boxes


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
        self.camera.add_tracklet_callback(lambda cam_id, tracklets, t: self.messages.append((cam_id, tracklets, t)))
        self.camera.start()

    def test_a_clip_frame_of_this_camera_is_stamped_and_handed_to_the_detector(self) -> None:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        before = time.perf_counter()
        self.player.push(0, image)          # another camera's frame
        self.player.push(1, image)
        self.assertEqual(len(self.frames), 1)
        self.assertEqual(self.frames[0].cam_id, 1)
        self.assertIs(self.frames[0].image, image)
        self.assertGreaterEqual(self.frames[0].timestamp, before)
        self.assertEqual(self.detector.submitted, [(1, image)])
        self.assertTrue(self.camera.connected)

    def test_the_detector_boxes_become_tracklets_with_the_message_time(self) -> None:
        before = time.perf_counter()
        self.detector.report(1, [Rect(0.1, 0.2, 0.3, 0.4), Rect(0.5, 0.5, 0.2, 0.3)])
        self.detector.report(0, [Rect(0.1, 0.2, 0.3, 0.4)])         # not this camera
        self.assertEqual(len(self.messages), 1)
        cam_id, tracklets, t = self.messages[0]
        self.assertEqual(cam_id, 1)
        self.assertEqual([tr.id for tr in tracklets], [0, 1])
        self.assertEqual(tracklets[0].roi, Rect(0.1, 0.2, 0.3, 0.4))
        self.assertEqual(tracklets[0].cam_id, 1)
        self.assertGreaterEqual(t, before)
        self.assertTrue(all(tr.timestamp == t for tr in tracklets))
        self.assertEqual(self.settings.readings.tracklets, 2)

    def test_nobody_is_a_message_too(self) -> None:
        self.detector.report(1, [])
        self.assertEqual(len(self.messages), 1)
        self.assertEqual(self.messages[0][1], [])

    def test_stopped_it_emits_nothing(self) -> None:
        self.camera.stop()
        self.player.push(1, np.zeros((4, 6, 3), dtype=np.uint8))
        self.detector.report(1, [Rect(0.1, 0.2, 0.3, 0.4)])
        self.assertEqual(self.frames, [])
        self.assertEqual(self.messages, [])
        self.assertFalse(self.camera.connected)


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
