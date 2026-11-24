"""An OAK-D hands frames over while it runs and none after `stop()`, as with its tracklets. Runs without a device, on
a depthai 2 install (`.venv`)."""

import unittest

import numpy as np

from modules.camera import CameraFrame, FrameType
from modules.camera.oak import Camera, OakCameraSettings

IMAGE: np.ndarray = np.zeros((4, 6), np.uint8)


class FrameStopTest(unittest.TestCase):
    def test_no_frame_after_stop(self) -> None:
        camera = Camera(OakCameraSettings())
        frames: list[CameraFrame] = []
        camera.add_frame_callback(frames.append)
        camera.running = True                               # as an open camera is
        camera._emit_frame(FrameType.VIDEO, IMAGE, 1.0, 1.0)
        camera.stop()
        camera._emit_frame(FrameType.VIDEO, IMAGE, 2.0, 2.0)
        self.assertEqual([frame.timestamp for frame in frames], [1.0])


if __name__ == '__main__':
    unittest.main()
