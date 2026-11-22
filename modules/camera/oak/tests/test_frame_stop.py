"""An OAK-D hands frames over while it runs and none after `stop()`, as with its tracklets; a passthrough clip camera
runs from its start. Runs without a device, on a depthai 2 install (`.venv`)."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from modules.camera import CameraFrame, FrameType
from modules.camera.oak import Camera, ClipCamera, OakCameraSettings

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

    def test_a_passthrough_clip_camera_runs_from_its_start_until_stop(self) -> None:
        settings = OakCameraSettings(clip_passthrough=True)
        camera = ClipCamera(MagicMock(), settings)
        frames: list[CameraFrame] = []
        camera.add_frame_callback(frames.append)
        camera.start()
        try:
            self.assertTrue(camera.connected)
            camera._passthrough_callback(0, FrameType.VIDEO, IMAGE, None, None, 1.0)
        finally:
            camera.stop()
            camera.join(1.0)
        camera._passthrough_callback(0, FrameType.VIDEO, IMAGE, None, None, 2.0)
        self.assertEqual([frame.timestamp for frame in frames], [1.0])


if __name__ == '__main__':
    unittest.main()
