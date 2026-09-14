"""Tests for the pose figure overlay's geometry: where a figure is drawn on the row's circle."""

import unittest

from apps.white_space.render.layers.pose_figure_layer import figure_spans


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
