"""Tests for the Vimba frame path against a fake frame (no camera, no thread started)."""

import unittest

import numpy as np

from modules.cam.settings import CameraSettings

try:
    from modules.cam.VimbaCam import VimbaCam
    from modules.cam.vimba import FrameStatus
except Exception:       # the Vimba library is not installed on this machine
    VimbaCam = None


class FakeFrame:
    def __init__(self, value: int, complete: bool = True) -> None:
        self.buffer = np.full((4, 6, 3), value, dtype=np.uint8)     # the driver's, reused
        self.complete = complete

    def get_status(self):
        return FrameStatus.Complete if self.complete else FrameStatus.Incomplete

    def as_numpy_ndarray(self) -> np.ndarray:
        return self.buffer


class FakeCamera:
    def __init__(self) -> None:
        self.queued: list[FakeFrame] = []

    def queue_frame(self, frame: FakeFrame) -> None:
        self.queued.append(frame)


@unittest.skipIf(VimbaCam is None, 'Vimba is not installed')
class TestVimbaCam(unittest.TestCase):
    def setUp(self):
        self.settings = CameraSettings()
        self.settings.initialize()
        self.cam = VimbaCam(self.settings, (6, 4))
        self.cam.capturing = True
        self.camera = FakeCamera()

    def tearDown(self):
        self.cam.close()

    def test_a_complete_frame_is_copied_and_given_back(self):
        frame = FakeFrame(10)
        self.cam.frameCallback(self.camera, frame)
        self.assertEqual(self.camera.queued, [frame])
        frame.buffer[:] = 99                                # the driver reuses the buffer
        taken = self.cam._take_internal_frame()
        self.assertEqual(int(taken[0, 0, 0]), 10)

    def test_a_frame_is_taken_once(self):
        self.cam.frameCallback(self.camera, FakeFrame(10))
        self.assertIsNotNone(self.cam._take_internal_frame())
        self.assertIsNone(self.cam._take_internal_frame())

    def test_an_incomplete_frame_is_skipped_counted_and_given_back(self):
        frame = FakeFrame(10, complete=False)
        self.cam.frameCallback(self.camera, frame)
        self.assertEqual(self.camera.queued, [frame])
        self.assertIsNone(self.cam._take_internal_frame())
        self.assertEqual(self.cam._features._dropped, 1)

    def test_no_frame_is_kept_while_not_capturing(self):
        self.cam.capturing = False
        frame = FakeFrame(10)
        self.cam.frameCallback(self.camera, frame)
        self.assertEqual(self.camera.queued, [frame])
        self.assertIsNone(self.cam._take_internal_frame())


if __name__ == '__main__':
    unittest.main()
