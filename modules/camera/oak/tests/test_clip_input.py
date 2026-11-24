"""An OAK-D fed a recording: a clip frame goes into the clip queue as the sensor would send it, planar BGR on a color
camera, grey on a mono one, stamped with its recorded capture; the camera hands only its own video frames to the open
device. Runs without a device (a fake queue), on a depthai 2 install (`.venv-oakd`)."""

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

    def test_each_frame_is_numbered_from_0(self) -> None:
        queue = FakeQueue()
        clip_input = ClipInput(queue, False, (6, 4), 0)
        for _ in range(3):
            clip_input.send(clip(bgr()))
        self.assertEqual([sent.getSequenceNum() for sent in queue.sent], [0, 1, 2])

    def test_a_gap_in_the_numbers_back_counts_the_frames_dropped(self) -> None:
        clip_input = ClipInput(FakeQueue(), False, (6, 4), 0)
        with self.assertLogs('modules.camera.oak.clip_input', 'INFO'):
            self.assertEqual(clip_input.returned(1), 1)            # number 0 never came back
        self.assertEqual(clip_input.returned(2), 0)
        self.assertEqual(clip_input.returned(5), 2)

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

    def test_the_camera_counts_the_clip_frames_dropped_since_the_start(self) -> None:
        camera = Camera(OakCameraSettings(), 0, clips=MagicMock())
        frame = dai.ImgFrame()
        frame.setType(dai.ImgFrame.Type.RAW8)
        frame.setWidth(6)
        frame.setHeight(4)
        frame.setData(np.zeros(24, np.uint8))  # pyright: ignore[reportCallIssue, reportArgumentType]
        for opening in range(2):                                    # each open numbers from 0 again
            camera.clip_input = ClipInput(FakeQueue(), False, (6, 4), 0)
            for sequence in (0, 2, 3):
                frame.setSequenceNum(sequence)
                camera._video_callback(frame)
        self.assertEqual(camera.readings.dropped_clips, 2)


if __name__ == '__main__':
    unittest.main()
