"""Tests for the webcam backend against a fake OpenCV capture (no device)."""

import time
import unittest
from unittest import mock

import numpy as np

from modules.camera import CameraFrame, CameraSettings
from modules.camera.webcam import Camera


class FakeCapture:
    """A capture that delivers one frame per read, 4 high and 6 wide, with a band across the top."""

    def __init__(self, opens: bool = True) -> None:
        self.opens = opens
        self.released = False
        self.reads = 0

    def isOpened(self) -> bool:
        return self.opens

    def read(self):
        self.reads += 1
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        image[0] = 255
        time.sleep(0.005)
        return True, image

    def release(self) -> None:
        self.released = True


def wait_for(condition, timeout: float = 2.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class TestWebCamera(unittest.TestCase):
    def setUp(self):
        self.settings = CameraSettings()
        self.settings.initialize()
        self.capture = FakeCapture()
        patcher = mock.patch('modules.camera.webcam.camera.cv2.VideoCapture', lambda index: self.capture)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.frames: list[CameraFrame] = []
        self.camera: Camera | None = None

    def tearDown(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.join()

    def test_delivers_frames_with_their_time_and_rate(self):
        self.camera = Camera(self.settings)
        self.camera.add_frame_callback(self.frames.append)
        before = time.perf_counter()
        self.camera.start()
        self.assertTrue(self.camera.connected)
        self.assertTrue(wait_for(lambda: len(self.frames) > 3))
        frame = self.frames[0]
        self.assertEqual(frame.cam_id, 0)
        self.assertEqual(frame.image.shape, (4, 6, 3))
        self.assertGreaterEqual(frame.time, before)
        self.assertGreater(self.settings.readings.video_fps, 0.0)
        self.assertGreater(self.settings.readings.video_min_fps, 0.0)

    def test_rotates_and_flips(self):
        settings = CameraSettings(flip_h=True)  # after the turn the band is a column on the left; mirrored, on the right
        settings.initialize()
        self.camera = Camera(settings, rotate90=True)
        self.camera.add_frame_callback(self.frames.append)
        self.camera.start()
        self.assertTrue(wait_for(lambda: bool(self.frames)))
        image = self.frames[0].image
        self.assertEqual(image.shape, (6, 4, 3))
        self.assertEqual(int(image[0, 0, 0]), 0)
        self.assertEqual(int(image[0, 3, 0]), 255)

    def test_stop_releases_the_device(self):
        self.camera = Camera(self.settings)
        self.camera.start()
        self.camera.stop()
        self.camera.join()
        self.assertTrue(self.capture.released)
        self.assertFalse(self.camera.connected)
        self.camera = None

    def test_a_device_that_does_not_open_is_a_warning(self):
        self.capture.opens = False
        self.camera = Camera(self.settings)
        with self.assertLogs('modules.camera.webcam.camera', 'WARNING'):
            self.camera.start()
        self.assertFalse(self.camera.connected)
        self.assertTrue(self.capture.released)
        self.camera.join()                      # never started: nothing to join
        self.camera = None


if __name__ == '__main__':
    unittest.main()
