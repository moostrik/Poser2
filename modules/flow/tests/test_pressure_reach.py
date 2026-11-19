"""pressure.reach as a share: the screened equation's 1/reach² for s from 0 to 1 of a field's longest side."""
import math
import unittest

from modules.flow.pressure_reach import screening


class TestPressureReach(unittest.TestCase):

    def test_everywhere_at_one(self) -> None:
        self.assertEqual(screening(1.0, 1.78), 0.0)

    def test_the_longest_side_at_one_half(self) -> None:
        self.assertAlmostEqual(screening(0.5, 1.78), 1.0 / 1.78 ** 2)

    def test_the_reach_is_longest_times_s_over_one_less_s(self) -> None:
        for share in (0.01, 0.1, 0.25, 0.75, 0.99):
            reach = 1.78 * share / (1.0 - share)
            self.assertAlmostEqual(screening(share, 1.78), 1.0 / reach ** 2, delta=1e-9 / reach ** 2)

    def test_shorter_as_s_falls(self) -> None:
        values = [screening(s / 100.0, 1.0) for s in range(1, 100)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        self.assertTrue(math.isfinite(values[0]))


if __name__ == '__main__':
    unittest.main()
