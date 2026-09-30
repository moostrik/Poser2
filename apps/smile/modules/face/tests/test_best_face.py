"""Tests for the best-face window: ranking, expiry and the size of the deque (no model)."""

import unittest

import numpy as np

from modules.face.BestFace import BestEntry, BestFace
from modules.face.FaceDetector import Face
from modules.face.FaceFinder import largest

IMAGE = np.zeros((4, 4, 3), dtype=np.uint8)


def face(x0=0.0, y0=0.0, x1=10.0, y1=10.0) -> Face:
    return Face(np.array([x0, y0, x1, y1]), 0.9, np.zeros((5, 2)))


def entry(time: float, accepted: bool, quality: float) -> BestEntry:
    return BestEntry(time, accepted, quality, IMAGE, face())


class TestBestFace(unittest.TestCase):
    def setUp(self):
        self.best = BestFace()

    def test_empty(self):
        self.assertIsNone(self.best.best())

    def test_highest_quality_wins(self):
        self.best.add(entry(0.0, True, 0.5), 1.0)
        self.best.add(entry(0.1, True, 0.8), 1.0)
        self.best.add(entry(0.2, True, 0.6), 1.0)
        self.assertEqual(self.best.best().quality, 0.8)

    def test_an_accepted_face_beats_any_rejected_one(self):
        self.best.add(entry(0.0, True, 0.3), 1.0)
        self.best.add(entry(0.1, False, 0.9), 1.0)
        self.assertTrue(self.best.best().accepted)

    def test_a_rejected_face_is_used_when_nothing_else_is_there(self):
        self.best.add(entry(0.0, False, 0.2), 1.0)
        self.best.add(entry(0.1, False, 0.4), 1.0)
        self.assertEqual(self.best.best().quality, 0.4)

    def test_the_best_expires_after_the_window(self):
        self.best.add(entry(0.0, True, 0.9), 1.0)
        self.best.add(entry(0.5, True, 0.6), 1.0)
        self.best.add(entry(1.5, True, 0.4), 1.0)
        self.assertEqual(self.best.best().quality, 0.6)

    def test_the_deque_holds_only_possible_winners(self):
        for i in range(100):
            self.best.add(entry(i * 0.01, True, 0.5), 1.0)
        self.assertEqual(len(self.best), 1)
        self.best.add(entry(1.0, True, 0.4), 1.0)
        self.assertEqual(len(self.best), 2)

    def test_the_newest_of_equal_faces_wins(self):
        self.best.add(entry(0.0, True, 0.5), 1.0)
        self.best.add(entry(0.5, True, 0.5), 1.0)
        self.assertEqual(self.best.best().time, 0.5)

    def test_clear(self):
        self.best.add(entry(0.0, True, 0.5), 1.0)
        self.best.clear()
        self.assertIsNone(self.best.best())


class TestLargest(unittest.TestCase):
    def test_largest_face(self):
        small, big = face(0, 0, 10, 10), face(20, 20, 50, 50)
        self.assertIs(largest([small, big]), big)
        self.assertIsNone(largest([]))


if __name__ == '__main__':
    unittest.main()
