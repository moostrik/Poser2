"""Tests for the YOLOv8 output decoding: the person class, the score threshold, the suppression
of overlapping boxes and the normalisation to the frame."""

import unittest

import numpy as np

from modules.inference.detection import person_boxes, nms, normalise


def output(candidates: list[tuple[float, float, float, float, float, float]], classes: int = 80) -> np.ndarray:
    """(4 + classes, N) from (cx, cy, w, h, person score, cat score) per candidate."""
    rows = np.zeros((4 + classes, len(candidates)), dtype=np.float32)
    for i, (cx, cy, w, h, person, cat) in enumerate(candidates):
        rows[0:4, i] = (cx, cy, w, h)
        rows[4, i] = person
        rows[4 + 15, i] = cat
    return rows


class PersonBoxesTest(unittest.TestCase):
    def test_only_people_above_the_threshold_are_boxes(self) -> None:
        out = output([(100, 100, 40, 80, 0.9, 0.0), (300, 100, 40, 80, 0.3, 0.0), (200, 200, 40, 40, 0.0, 0.95)])
        boxes, scores = person_boxes(out, confidence=0.5, overlap=0.5)
        self.assertEqual(len(boxes), 1)
        np.testing.assert_allclose(boxes[0], (80, 60, 120, 140))
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)

    def test_overlapping_boxes_keep_the_best_one(self) -> None:
        out = output([(100, 100, 40, 80, 0.7, 0.0), (102, 101, 40, 80, 0.9, 0.0), (400, 100, 40, 80, 0.6, 0.0)])
        boxes, scores = person_boxes(out, confidence=0.5, overlap=0.5)
        self.assertEqual(len(boxes), 2)
        np.testing.assert_allclose(scores, (0.9, 0.6), rtol=1e-5)      # by score, the duplicate gone
        np.testing.assert_allclose(boxes[0], (82, 61, 122, 141))

    def test_nothing_gives_empty_arrays(self) -> None:
        boxes, scores = person_boxes(output([(100, 100, 40, 80, 0.1, 0.0)]), confidence=0.5, overlap=0.5)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))


class NmsTest(unittest.TestCase):
    def test_a_box_inside_another_is_suppressed_at_any_overlap_above_the_limit(self) -> None:
        boxes = np.array([[0, 0, 100, 100], [10, 10, 90, 90], [200, 200, 300, 300]], dtype=np.float32)
        scores = np.array([0.8, 0.9, 0.5], dtype=np.float32)
        self.assertEqual(list(nms(boxes, scores, overlap=0.5)), [1, 2])     # 0.64 overlap: gone
        self.assertEqual(list(nms(boxes, scores, overlap=0.7)), [1, 0, 2])  # kept below 0.7


class NormaliseTest(unittest.TestCase):
    def test_boxes_become_fractions_of_the_input_and_are_clipped(self) -> None:
        boxes = np.array([[128, 112, 256, 448], [-10, 0, 20, 500]], dtype=np.float32)
        rects = normalise(boxes, width=512, height=448)
        np.testing.assert_allclose(rects[0], (0.25, 0.25, 0.25, 0.75))
        np.testing.assert_allclose(rects[1], (0.0, 0.0, 20 / 512, 1.0))

    def test_no_boxes(self) -> None:
        self.assertEqual(normalise(np.empty((0, 4)), 512, 448).shape, (0, 4))


if __name__ == '__main__':
    unittest.main()
