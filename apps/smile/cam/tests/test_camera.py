"""Tests for the camera front: the image source against a fake device (no camera, no GL)."""

import os
import tempfile
import threading
import time
import unittest

import cv2
import numpy as np

from apps.smile.cam.camera import Camera, list_images
from apps.smile.cam.settings import CameraSettings
from modules.camera import CameraFrame, FrameType


class FakeDevice:
    """A `CameraSource` whose connection the test switches; the test pushes its frames."""

    def __init__(self, connects: bool = True) -> None:
        self.connects = connects
        self.callbacks = []
        self.started = False
        self.stopped = False
        self.joined = False

    @property
    def connected(self) -> bool:
        return self.connects

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout=None) -> None:
        self.joined = True

    def add_frame_callback(self, callback) -> None:
        self.callbacks.append(callback)

    def push(self, image: np.ndarray, rotate90: bool = False, flip_h: bool = False) -> None:
        frame = CameraFrame(cam_id=0, kind=FrameType.VIDEO, image=image, timestamp=time.perf_counter(),
                            rotate90=rotate90, flip_h=flip_h)
        for c in self.callbacks:
            c(frame)


def wait_for(condition, timeout: float = 1.0) -> bool:
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class TestCamera(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        for name, value in (('b.jpg', 200), ('a.jpg', 100)):
            cv2.imwrite(os.path.join(self.folder, name), np.full((30, 20, 3), value, dtype=np.uint8))
        self.settings = CameraSettings()
        self.settings.initialize()
        self.frames: list[CameraFrame] = []
        self.camera: Camera | None = None

    def tearDown(self):
        if self.camera is not None:
            self.camera.stop()

    def make(self, device):
        self.camera = Camera(self.settings, device, 'fake', folder=self.folder)
        self.camera.add_frame_callback(self.frames.append)
        self.camera.start()
        return self.camera

    def test_lists_the_folder_and_selects_the_first_image(self):
        self.make(FakeDevice())
        self.assertEqual(self.settings.image_options, ['a.jpg', 'b.jpg'])
        self.assertEqual(self.settings.image_count, 2)
        self.assertEqual(self.settings.image, 'a.jpg')
        self.assertEqual(self.settings.device, 'fake')

    def test_relays_device_frames_and_drops_them_with_the_image_on(self):
        device = FakeDevice()
        self.make(device)
        frame = np.zeros((12, 8, 3), dtype=np.uint8)
        device.push(frame)
        self.assertTrue(wait_for(lambda: len(self.frames) == 1))     # handed on whole by the camera's own thread
        shown = self.camera.frame().image
        self.assertIs(shown, frame)

        self.settings.use_image = True
        device.push(frame)
        self.assertTrue(wait_for(lambda: len(self.frames) > 1 and self.frames[-1].image is not shown))
        image = self.camera.frame().image
        self.assertEqual(image.shape, (30, 20, 3))
        self.assertEqual(int(image[0, 0, 0]), 100)     # a.jpg, whole

    def test_a_turned_frame_is_handed_on_whole_with_its_flags(self):
        device = FakeDevice()
        self.make(device)
        device.push(np.zeros((16, 12, 3), dtype=np.uint8), rotate90=True, flip_h=True)
        self.assertTrue(wait_for(lambda: len(self.frames) == 1))
        frame = self.frames[0]
        self.assertEqual(frame.image.shape, (16, 12, 3))
        self.assertTrue(frame.rotate90 and frame.flip_h)

    def test_every_device_frame_goes_to_the_recorder_as_it_came(self):
        device = FakeDevice()
        recorded: list[CameraFrame] = []
        self.make(device)
        self.camera.add_device_frame_callback(recorded.append)
        image = np.zeros((16, 12, 3), dtype=np.uint8)
        device.push(image, rotate90=True)
        self.settings.use_image = True          # the live path shows the image: the device is still recorded
        device.push(image, rotate90=True)
        self.assertEqual(len(recorded), 2)
        self.assertTrue(all(frame.image is image and frame.rotate90 for frame in recorded))

    def test_a_frame_has_the_time_it_arrived(self):
        device = FakeDevice()
        self.make(device)
        self.assertEqual(self.camera.frame().timestamp, 0.0)      # no frame yet
        self.assertTrue(self.camera.stalled)
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        before = time.perf_counter()
        device.push(frame)
        self.assertTrue(wait_for(lambda: self.camera.frame().timestamp != 0.0))
        shown = self.camera.frame()
        self.assertIs(shown.image, frame)
        self.assertGreaterEqual(shown.timestamp, before)
        self.assertFalse(self.camera.stalled)
        # a camera that stops keeps its last frame, with the time of that frame
        self.assertEqual(self.camera.frame().timestamp, shown.timestamp)

    def test_device_and_phone_frames_are_handed_on_from_the_cameras_thread(self):
        threads: list[str] = []
        device = FakeDevice()
        self.make(device)
        self.camera.add_frame_callback(lambda _frame: threads.append(threading.current_thread().name))
        device.push(np.zeros((12, 8, 3), dtype=np.uint8))
        self.assertTrue(wait_for(lambda: len(self.frames) == 1))
        self.camera.set_online(True)
        self.camera.submit_online_frame(np.zeros((24, 16, 3), dtype=np.uint8))
        self.assertTrue(wait_for(lambda: len(self.frames) == 2))
        self.assertEqual(self.frames[1].image.shape, (24, 16, 3))
        self.assertEqual(threads, ['SmileCamera', 'SmileCamera'])

    def test_a_failing_callback_is_logged_and_the_next_one_still_called(self):
        device = FakeDevice()
        self.make(device)

        def fail(_frame) -> None:
            raise RuntimeError('a receiver failed')
        after: list[CameraFrame] = []
        self.camera.clear_frame_callbacks()
        self.camera.add_frame_callback(fail)                # first: the one after it still gets the frame
        self.camera.add_frame_callback(after.append)
        with self.assertLogs('modules.utils.broadcast', level='ERROR'):
            device.push(np.zeros((6, 8, 3), dtype=np.uint8))
            self.assertTrue(wait_for(lambda: len(after) == 1))
        device.push(np.zeros((6, 8, 3), dtype=np.uint8))
        self.assertTrue(wait_for(lambda: len(after) == 2))

    def test_a_missing_device_is_none_and_the_image_still_works(self):
        self.make(FakeDevice(connects=False))
        self.assertEqual(self.settings.device, 'none')
        self.assertEqual(self.frames, [])
        self.settings.use_image = True
        self.assertTrue(wait_for(lambda: len(self.frames) > 0))

    def test_a_device_that_connects_later_delivers_and_shows(self):
        device = FakeDevice(connects=False)
        self.make(device)
        self.assertTrue(device.started)         # kept: it keeps trying
        self.assertEqual(self.settings.device, 'none')
        device.connects = True                  # found after all
        self.assertTrue(wait_for(lambda: self.settings.device == 'fake'))
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        device.push(frame)
        self.assertTrue(wait_for(lambda: self.camera.frame().image is frame))
        device.connects = False                 # and gone again
        self.assertTrue(wait_for(lambda: self.settings.device == 'none'))

    def test_changing_the_image_loads_the_new_file(self):
        self.make(FakeDevice())
        self.settings.use_image = True
        self.settings.image = 'b.jpg'
        self.assertTrue(wait_for(lambda: self.frames and int(self.frames[-1].image[0, 0, 0]) == 200))

    def test_reload_picks_up_a_new_file_and_keeps_the_selection(self):
        self.make(FakeDevice())
        self.settings.image = 'b.jpg'
        cv2.imwrite(os.path.join(self.folder, 'c.jpg'), np.zeros((4, 4, 3), dtype=np.uint8))
        self.settings.reload = True
        self.assertEqual(self.settings.image_options, ['a.jpg', 'b.jpg', 'c.jpg'])
        self.assertEqual(self.settings.image_count, 3)
        self.assertEqual(self.settings.image, 'b.jpg')

    def test_next_image_walks_the_folder_and_wraps(self):
        self.make(FakeDevice())
        self.settings.use_image = True
        self.settings.image_next = True
        self.assertEqual(self.settings.image, 'a.jpg')
        self.camera.next_image()
        self.assertEqual(self.settings.image, 'b.jpg')
        self.camera.next_image()
        self.assertEqual(self.settings.image, 'a.jpg')

    def test_next_image_does_nothing_without_the_toggles(self):
        self.make(FakeDevice())
        self.settings.use_image = True
        self.camera.next_image()
        self.assertEqual(self.settings.image, 'a.jpg')
        self.settings.use_image = False
        self.settings.image_next = True
        self.camera.next_image()
        self.assertEqual(self.settings.image, 'a.jpg')

    def test_stop_stops_the_device(self):
        device = FakeDevice()
        self.make(device)
        self.camera.stop()
        self.camera = None
        self.assertTrue(device.stopped and device.joined)


class TestHelpers(unittest.TestCase):
    def test_list_images_sorted_and_missing_folder(self):
        self.assertEqual(list_images(os.path.join(tempfile.gettempdir(), 'no-such-folder-smile')), [])


if __name__ == '__main__':
    unittest.main()
