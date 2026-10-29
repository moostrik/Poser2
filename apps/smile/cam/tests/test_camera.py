"""Tests for the camera front: the image source against a fake device (no camera, no GL)."""

import os
import tempfile
import time
import unittest

import cv2
import numpy as np

from apps.smile.cam.camera import Camera, cover, list_images
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

    def push(self, image: np.ndarray) -> None:
        frame = CameraFrame(cam_id=0, kind=FrameType.VIDEO, image=image, timestamp=time.perf_counter())
        for c in self.callbacks:
            c(frame)


def wait_for(condition, timeout: float = 1.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
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
        self.frames: list[np.ndarray] = []
        self.camera: Camera | None = None

    def tearDown(self):
        if self.camera is not None:
            self.camera.stop()

    def make(self, device):
        self.camera = Camera(self.settings, (8, 6), device, 'fake', folder=self.folder)
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
        # a frame of another shape is fitted to the output size: scaled to fill it, the middle kept
        frame = np.zeros((12, 8, 3), dtype=np.uint8)
        frame[4:8] = 255                    # a band across the middle
        device.push(frame)
        self.assertEqual(len(self.frames), 1)
        shown = self.camera.frame()[0]
        self.assertEqual(shown.shape, (6, 8, 3))
        self.assertEqual(int(shown[0, 0, 0]), 0)
        self.assertEqual(int(shown[3, 0, 0]), 255)

        self.settings.use_image = True
        device.push(frame)
        self.assertTrue(wait_for(lambda: len(self.frames) > 1 and self.frames[-1] is not shown))
        image = self.camera.frame()[0]
        self.assertEqual(image.shape, (6, 8, 3))
        self.assertEqual(int(image[0, 0, 0]), 100)     # a.jpg, at the output size

    def test_a_frame_has_the_time_it_arrived(self):
        device = FakeDevice()
        self.make(device)
        self.assertEqual(self.camera.frame()[1], 0.0)      # no frame yet
        self.assertTrue(self.camera.stalled)
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        before = time.perf_counter()
        device.push(frame)
        image, arrived = self.camera.frame()
        self.assertIs(image, frame)
        self.assertGreaterEqual(arrived, before)
        self.assertFalse(self.camera.stalled)
        # a camera that stops keeps its last frame, with the time of that frame
        self.assertEqual(self.camera.frame()[1], arrived)

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
        self.assertIs(self.camera.frame()[0], frame)
        device.connects = False                 # and gone again
        self.assertTrue(wait_for(lambda: self.settings.device == 'none'))

    def test_changing_the_image_loads_the_new_file(self):
        self.make(FakeDevice())
        self.settings.use_image = True
        self.settings.image = 'b.jpg'
        self.assertTrue(wait_for(lambda: self.frames and int(self.frames[-1][0, 0, 0]) == 200))

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
    def test_cover_keeps_the_aspect_and_fills_the_size(self):
        wide = np.zeros((10, 40, 3), dtype=np.uint8)
        self.assertEqual(cover(wide, 8, 6).shape, (6, 8, 3))
        tall = np.zeros((40, 10, 3), dtype=np.uint8)
        self.assertEqual(cover(tall, 8, 6).shape, (6, 8, 3))

    def test_list_images_sorted_and_missing_folder(self):
        self.assertEqual(list_images(os.path.join(tempfile.gettempdir(), 'no-such-folder-smile')), [])


if __name__ == '__main__':
    unittest.main()
