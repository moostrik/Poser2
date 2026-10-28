"""Tests for the face finder's hand-over of camera frames (a fake detector, no model, no thread)."""

import unittest

import numpy as np

from apps.smile.face.FaceFinder import FaceFinder, FaceFrame
from apps.smile.face.settings import FaceSettings


class NoFaces:
    def detect(self, image: np.ndarray, threshold: float) -> list:
        return []


class TestFaceFinder(unittest.TestCase):
    def setUp(self):
        self.finder = FaceFinder(FaceSettings(), 100, 125)
        self.frames: list[FaceFrame] = []
        self.finder.add_frame_callback(self.frames.append)
        self.image = np.zeros((125, 100, 3), dtype=np.uint8)

    def tearDown(self):
        self.finder.stop()

    def test_a_frame_is_taken_once(self):
        self.finder.set_image(self.image, 12.5)
        image, arrived = self.finder._take_image()
        self.assertIs(image, self.image)
        self.assertEqual(arrived, 12.5)
        self.assertIsNone(self.finder._take_image()[0])

    def test_the_faces_have_the_time_of_their_camera_frame(self):
        # not the time of the detection: a camera that stopped must go stale
        self.finder._process(self.image, 12.5, NoFaces(), None)
        self.assertEqual(len(self.frames), 1)
        self.assertEqual(self.frames[0].time, 12.5)
        self.assertFalse(self.frames[0].qualified)


if __name__ == '__main__':
    unittest.main()
