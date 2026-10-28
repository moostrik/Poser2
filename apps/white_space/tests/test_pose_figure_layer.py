"""Tests for the pose figure overlay's geometry: where a figure is drawn on the row's circle."""

import math
import unittest

import numpy as np

from modules.pose.features import Points2D, PointLandmark as P

from apps.white_space.render.layers.pose_figure_layer import eye_column, figure_spans


def _points(**xy: tuple[float, float]) -> Points2D:
    values = np.full((len(P), 2), np.nan, dtype=np.float32)
    for name, (x, y) in xy.items():
        values[P[name]] = (x, y)
    return Points2D(values, (~np.isnan(values[:, 0])).astype(np.float32))


class EyeColumnTest(unittest.TestCase):
    def test_the_eyes_mean_when_both_are_there(self) -> None:
        self.assertAlmostEqual(eye_column(_points(left_eye=(0.7, 0.1), right_eye=(0.6, 0.1), nose=(0.2, 0.2))), 0.65, places=6)

    def test_the_nose_when_an_eye_is_missing(self) -> None:
        self.assertAlmostEqual(eye_column(_points(left_eye=(0.7, 0.1), nose=(0.62, 0.2))), 0.62, places=6)

    def test_the_centre_when_the_face_is_missing(self) -> None:
        self.assertEqual(eye_column(_points(left_shoulder=(0.3, 0.3))), 0.5)


class FigureSpansTest(unittest.TestCase):
    def test_a_figure_inside_the_row_is_drawn_once(self) -> None:
        self.assertEqual(figure_spans(0.4, 0.1), [0.4])
        self.assertEqual(figure_spans(0.0, 0.1), [0.0])
        self.assertEqual(figure_spans(0.9, 0.1), [0.9])

    def test_a_figure_over_the_join_is_drawn_on_both_sides(self) -> None:
        for x, expected in ((-0.05, [-0.05, 0.95]), (0.95, [0.95, -0.05])):
            spans = figure_spans(x, 0.1)
            self.assertEqual(len(spans), 2)
            for span, want in zip(spans, expected):
                self.assertAlmostEqual(span, want, places=9)


if __name__ == "__main__":
    unittest.main()
