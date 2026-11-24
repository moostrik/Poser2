"""An OAK-D fed a recording: a clip frame goes into the clip queue as the sensor would send it, planar BGR on a color
camera, grey on a mono one, stamped with its recorded capture; the camera hands only its own video frames to the open
device. Runs without a device (a fake queue), on a depthai 2 install (`.venv`)."""

import unittest
from time import perf_counter
from unittest.mock import MagicMock

import depthai as dai
import numpy as np

from modules.camera import ClipFrame, FrameType
from modules.camera.oak import Camera, OakCameraSettings
from modules.camera.oak.clip_input import ClipInput


class FakeQueue:
    def __init__(self) -> None:
        self.sent: list[dai.ImgFrame] = []

    def send(self, msg: dai.ImgFrame) -> None:
        self.sent.append(msg)


def bgr(height: int = 4, width: int = 6) -> np.ndarray:
    image = np.zeros((height, width, 3), np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 10, 20, 30           # B, G, R
    return image


def clip(image: np.ndarray, cam_id: int = 0, kind: FrameType = FrameType.VIDEO, timestamp: float | None = 12.5) -> ClipFrame:
    return ClipFrame(cam_id, kind, image, None, None, timestamp)


class ClipInputTest(unittest.TestCase):
    def test_a_color_frame_goes_in_planar_bgr_stamped_with_its_recorded_capture(self) -> None:
        queue = FakeQueue()
        ClipInput(queue, True, (6, 4), 0).send(clip(bgr()))
        sent = queue.sent[0]
        self.assertEqual(sent.getType(), dai.ImgFrame.Type.BGR888p)
        self.assertEqual((sent.getWidth(), sent.getHeight()), (6, 4))
        data = np.asarray(sent.getData())
        np.testing.assert_array_equal(data[:24], 10)              # the B plane, then G, then R
        np.testing.assert_array_equal(data[24:48], 20)
        np.testing.assert_array_equal(data[48:], 30)
        self.assertAlmostEqual(sent.getTimestamp().total_seconds(), 12.5, places=6)

    def test_a_mono_camera_takes_it_grey(self) -> None:
        queue = FakeQueue()
        ClipInput(queue, False, (6, 4), 0).send(clip(bgr()))
        sent = queue.sent[0]
        self.assertEqual(sent.getType(), dai.ImgFrame.Type.RAW8)
        self.assertEqual(len(sent.getData()), 24)

    def test_a_frame_without_a_recorded_capture_is_stamped_at_its_send(self) -> None:
        queue = FakeQueue()
        before: float = perf_counter()
        ClipInput(queue, False, (6, 4), 0).send(clip(bgr(), timestamp=None))
        self.assertGreaterEqual(queue.sent[0].getTimestamp().total_seconds(), before - 1e-6)

    def test_a_frame_of_another_size_warns_once(self) -> None:
        queue = FakeQueue()
        clip_input = ClipInput(queue, False, (8, 8), 0)
        with self.assertLogs('modules.camera.oak.clip_input', 'WARNING') as logs:
            clip_input.send(clip(bgr()))
            clip_input.send(clip(bgr()))
        self.assertEqual(len(logs.records), 1)


class CameraClipsTest(unittest.TestCase):
    def test_the_camera_hands_only_its_own_video_frames_to_the_open_device(self) -> None:
        player = MagicMock()
        camera = Camera(OakCameraSettings(), 1, clips=player)
        clip_input = MagicMock()
        camera._clip_callback(clip(bgr(), cam_id=1))              # closed: nothing to send it to
        camera.clip_input, camera.running = clip_input, True       # as an open camera is
        camera._clip_callback(clip(bgr(), cam_id=0))
        camera._clip_callback(clip(bgr(), cam_id=1, kind=FrameType.DEPTH))
        camera._clip_callback(clip(bgr(), cam_id=1))
        self.assertEqual(clip_input.send.call_count, 1)


if __name__ == '__main__':
    unittest.main()
