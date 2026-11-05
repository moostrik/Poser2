"""The log axes of the monitor's strip and histogram: the heights `strip_roll.frag` and
`histogram_draw.frag` draw a value at, and the labels the overlay puts there."""

import math
import unittest

from modules.flow.visualization.log_axis import LANE_FILL, value_limit, histogram_lines, strip_height, strip_lines


class StripLinesTest(unittest.TestCase):
    def test_each_line_sits_where_the_strip_draws_its_label(self) -> None:
        for scale in (0.5, 1.07, 5.0, 37.0):
            for line in strip_lines(scale):
                self.assertAlmostEqual(strip_height(float(line.label), scale), line.fraction, places=9)

    def test_lines_are_round_decades_inside_the_lane(self) -> None:
        self.assertEqual([line.label for line in strip_lines(1.07)], ['0.01', '0.1'])
        self.assertEqual([line.label for line in strip_lines(5.0)], ['0.001', '0.01', '0.1'])
        self.assertEqual([line.label for line in strip_lines(1.0)], ['0.01', '0.1', '1'])
        for line in strip_lines(5.0):
            self.assertLessEqual(line.fraction, LANE_FILL)

    def test_no_scale_has_no_lines(self) -> None:
        self.assertEqual(strip_lines(0.0), [])


class HistogramLinesTest(unittest.TestCase):
    def test_round_shares_of_the_samples(self) -> None:
        lines = histogram_lines(65536)
        self.assertEqual([line.label for line in lines], ['0.01%', '0.1%', '1%', '10%'])
        fractions = [line.fraction for line in lines]
        self.assertEqual(fractions, sorted(fractions))
        self.assertLess(fractions[-1], LANE_FILL)

    def test_a_line_sits_where_a_bin_of_its_share_draws(self) -> None:
        samples = 65536
        line = histogram_lines(samples)[2]                     # 1%
        count = 0.01 * samples
        self.assertAlmostEqual(line.fraction, LANE_FILL * math.log1p(count) / math.log1p(samples))

    def test_no_line_under_one_sample(self) -> None:
        self.assertEqual([line.label for line in histogram_lines(500)], ['1%', '10%'])

    def test_the_limit_is_one_over_scale(self) -> None:
        self.assertEqual(value_limit(4.0), '0.25')
        self.assertEqual(value_limit(0.0), '-')


if __name__ == '__main__':
    unittest.main()
