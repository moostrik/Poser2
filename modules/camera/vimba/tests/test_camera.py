"""Tests for the Vimba frame path against a fake frame (no camera, no thread started), and for the
thread that opens the camera again against a fake camera (no Vimba system started). They need the
binding, so they run where the Vimba SDK is installed."""

import contextlib
import time
import unittest
from unittest import mock

import numpy as np

from modules.camera import CameraFrame
from modules.camera.vimba import Camera, VimbaCameraSettings
from modules.camera.vimba.camera import CameraSetup, NoCamera, VIMBA_ERROR

if VIMBA_ERROR is None:
    from modules.camera.vimba.vimbapython import FrameStatus


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


@unittest.skipIf(VIMBA_ERROR is not None, 'Vimba is not installed')
class TestVimbaCam(unittest.TestCase):
    def setUp(self):
        self.settings = VimbaCameraSettings()
        self.settings.initialize()
        self.cam = Camera(self.settings)
        self.cam.running = True
        self.camera = FakeCamera()

    def tearDown(self):
        self.cam.stop()
        self.cam.join()

    def test_a_complete_frame_is_copied_and_given_back(self):
        frame = FakeFrame(10)
        self.cam._frame_handler(self.camera, frame)
        self.assertEqual(self.camera.queued, [frame])
        frame.buffer[:] = 99                                # the driver reuses the buffer
        taken = self.cam._take_frame()
        self.assertEqual(int(taken[0, 0, 0]), 10)

    def test_a_frame_is_taken_once(self):
        self.cam._frame_handler(self.camera, FakeFrame(10))
        self.assertIsNotNone(self.cam._take_frame())
        self.assertIsNone(self.cam._take_frame())

    def test_an_incomplete_frame_is_skipped_counted_and_given_back(self):
        frame = FakeFrame(10, complete=False)
        self.cam._frame_handler(self.camera, frame)
        self.assertEqual(self.camera.queued, [frame])
        self.assertIsNone(self.cam._take_frame())
        self.assertEqual(self.cam.control._dropped, 1)

    def test_no_frame_is_kept_while_stopped(self):
        self.cam.running = False
        frame = FakeFrame(10)
        self.cam._frame_handler(self.camera, frame)
        self.assertEqual(self.camera.queued, [frame])
        self.assertIsNone(self.cam._take_frame())


class StreamingCamera(FakeCamera):
    """A camera the thread can open and stream from; the test pushes its frames."""

    def __init__(self) -> None:
        super().__init__()
        self.handler = None
        self.streams: int = 0

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def get_id(self) -> str:
        return 'fake'

    def start_streaming(self, handler, **kwargs) -> None:
        self.handler = handler
        self.streams += 1

    def stop_streaming(self) -> None:
        self.handler = None


def wait_for(condition, timeout: float = 3.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@unittest.skipIf(VIMBA_ERROR is not None, 'Vimba is not installed')
class TestVimbaCamReopen(unittest.TestCase):
    def setUp(self):
        self.settings = VimbaCameraSettings()
        self.settings.initialize()
        self.cam = Camera(self.settings)
        self.cam.control.update = lambda camera: None       # no features on the fake camera
        self.frames: list[CameraFrame] = []
        self.cam.add_frame_callback(self.frames.append)
        self.camera = StreamingCamera()
        self.found: bool = True
        self.searches: int = 0

        def get_camera(camera_id):
            self.searches += 1
            if not self.found:
                raise NoCamera('no camera found')
            return self.camera

        vimba = mock.Mock()
        vimba.get_instance.return_value = contextlib.nullcontext()
        setup = CameraSetup(1500, 1200, 960, 46, 2)
        for name, value in (('Vimba', vimba), ('get_camera', get_camera), ('setup_camera', lambda camera, user_set: setup),
                            ('FRAME_TIMEOUT', 0.1), ('RETRY_INTERVAL', 0.02), ('STREAM_PAUSE', 0.02)):
            patcher = mock.patch(f'modules.camera.vimba.camera.{name}', value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.cam.stop()
        self.cam.join()

    def push(self) -> bool:
        handler = self.camera.handler
        if handler is not None:
            handler(self.camera, FakeFrame(10))
        return bool(self.frames)

    def test_a_camera_that_is_found_later_delivers(self):
        self.found = False
        self.cam.start()
        self.assertFalse(self.cam.connected)
        self.assertTrue(wait_for(lambda: self.searches > 2))        # it keeps looking
        self.assertFalse(self.cam.connected)
        self.found = True
        self.assertTrue(wait_for(self.push))
        self.assertTrue(self.cam.connected)
        self.assertEqual(self.frames[0].image.shape, (4, 6, 3))
        self.assertGreater(self.frames[0].timestamp, 0.0)

    def test_a_camera_whose_frames_stop_is_opened_again(self):
        self.cam.start()
        self.assertTrue(self.cam.connected)
        self.assertTrue(wait_for(self.push))
        self.assertEqual((self.searches, self.camera.streams), (1, 1))
        # no frames any more: searched for and streamed again, and the panel's values are sent again
        with mock.patch.object(self.cam.control, 'reset') as reset:
            self.assertTrue(wait_for(lambda: self.camera.streams > 1))
            reset.assert_called()
        self.frames.clear()
        self.assertTrue(wait_for(self.push))

    def test_frames_that_keep_coming_keep_the_stream(self):
        self.cam.start()
        end = time.time() + 0.4                                     # four times the frame timeout
        while time.time() < end:
            self.push()
            time.sleep(0.01)
        self.assertEqual((self.searches, self.camera.streams), (1, 1))


if __name__ == '__main__':
    unittest.main()
