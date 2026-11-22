"""The flags a frame carries: the turn the app asked for, and the flips the camera leaves to the host; a turned
camera sends no tilt homography. Runs without a device. On a depthai 3 install (`.venv-oak4`)."""

import unittest

import numpy as np

from modules.camera import CameraFrame, FrameType
from modules.camera.oak4 import Oak4CameraSettings
from modules.camera.oak4.camera import Camera

CAMERA: str = 'modules.camera.oak4.camera'


def emitted(camera: Camera) -> CameraFrame:
    frames: list[CameraFrame] = []
    camera.running = True                               # as an open camera is
    camera.add_frame_callback(frames.append)
    camera._emit_frame(FrameType.VIDEO, np.zeros((6, 4), np.uint8), 1.0, 1.0, nv12=True)
    return frames[0]


class FrameFlagsTest(unittest.TestCase):
    def test_a_frame_carries_the_turn_and_both_flips(self) -> None:
        camera = Camera(Oak4CameraSettings(rotate90=True))
        camera.flip_h_on_host, camera.flip_v_on_host = True, True
        frame = emitted(camera)
        self.assertEqual((frame.rotate90, frame.flip_v, frame.flip_h), (True, True, True))

    def test_an_unturned_camera_sends_no_turn(self) -> None:
        self.assertFalse(emitted(Camera(Oak4CameraSettings())).rotate90)

    def test_a_turned_camera_sends_no_homography_and_warns_once_with_tilt_on(self) -> None:
        settings = Oak4CameraSettings(rotate90=True)
        settings.tilt.enabled = True
        camera = Camera(settings)
        camera.tilt_correction.homography = np.eye(3)       # as a tilted camera would have one
        with self.assertLogs(CAMERA, 'WARNING') as logs:
            self.assertIsNone(emitted(camera).homography)
            camera._emit_frame(FrameType.VIDEO, np.zeros((6, 4), np.uint8), 2.0, 2.0, nv12=True)
        self.assertEqual(len(logs.records), 1)

    def test_no_frame_after_stop(self) -> None:
        camera = Camera(Oak4CameraSettings())
        emitted(camera)
        frames: list[CameraFrame] = []
        camera.add_frame_callback(frames.append)
        camera.stop()
        camera._emit_frame(FrameType.VIDEO, np.zeros((6, 4), np.uint8), 3.0, 3.0, nv12=True)
        self.assertEqual(frames, [])


if __name__ == '__main__':
    unittest.main()
