"""Tests for the log slider's position <-> value mapping."""

import unittest

from modules.settings import Widget


class TestLogSliderMapping(unittest.TestCase):

    def test_endpoints_map_to_min_and_max(self):
        self.assertEqual(Widget.log_value(Widget.log_position(0.001)), 0.001)
        self.assertEqual(Widget.log_value(Widget.log_position(100.0)), 100.0)

    def test_midpoint_is_geometric_mean(self):
        low, high = Widget.log_position(0.01), Widget.log_position(100.0)
        self.assertEqual(Widget.log_value((low + high) / 2), 1.0)

    def test_round_trip_keeps_three_significant_digits(self):
        for value in (0.00123, 0.025, 0.7, 1.5, 12.3, 99.9):
            self.assertEqual(Widget.log_value(Widget.log_position(value)), value)

    def test_value_is_rounded_to_three_significant_digits(self):
        self.assertEqual(Widget.log_value(Widget.log_position(0.123456)), 0.123)

    def test_every_decade_takes_equal_slider_travel(self):
        travel = [Widget.log_position(v * 10) - Widget.log_position(v) for v in (0.001, 0.1, 10.0)]
        for t in travel:
            self.assertAlmostEqual(t, 1.0)

    def test_arrow_step_is_a_tenth_of_the_decade(self):
        for value, step in ((0.0025, 0.0001), (0.025, 0.001), (0.7, 0.01), (1.0, 0.1), (5.0, 0.1), (19.0, 1.0)):
            self.assertEqual(Widget.log_step(value), step)

    def test_arrow_step_never_reaches_zero(self):
        for value in (0.001, 0.0999, 0.1, 9.99, 100.0):
            self.assertLess(Widget.log_step(value), value)

    def test_log_slider_accepts_numeric_fields_only(self):
        self.assertTrue(Widget.log_slider.accepts(float))
        self.assertTrue(Widget.log_slider.accepts(int))
        self.assertFalse(Widget.log_slider.accepts(bool))
        self.assertFalse(Widget.log_slider.accepts(str))


if __name__ == "__main__":
    unittest.main()
