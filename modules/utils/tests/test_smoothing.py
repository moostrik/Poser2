"""The single-value filters: the Ema's time constant, its rise and fall, the rpm's conversion, the One Euro filter,
and an angle filtered across ±π."""

import math
import unittest

from modules.utils import AngleFilter, EmaFilter, OneEuroFilter


def run(f, values: list[float], timestamps: list[float]) -> float:
    """The filter's value after each value at its timestamp."""
    result: float = math.nan
    for value, timestamp in zip(values, timestamps):
        result = f(value, timestamp)
    return result


class EmaFilterTest(unittest.TestCase):
    def test_a_step_reaches_63_percent_after_one_time_constant_at_any_rate(self) -> None:
        for rate in (10.0, 60.0, 1000.0):
            ema = EmaFilter(0.5)
            ema(0.0, 0.0)
            steps: int = int(0.5 * rate)
            value: float = run(ema, [1.0] * steps, [(i + 1) / rate for i in range(steps)])
            self.assertAlmostEqual(value, 1.0 - math.exp(-1.0), places=9, msg=f'at {rate} Hz')

    def test_a_fall_follows_its_own_time_constant(self) -> None:
        ema = EmaFilter(0.1, fall_time_constant=1.0)
        ema(0.0, 0.0)
        self.assertAlmostEqual(ema(1.0, 0.1), 1.0 - math.exp(-1.0))          # a rise at 0.1 s
        ema.reset(1.0)
        ema(1.0, 0.0)
        self.assertAlmostEqual(ema(0.0, 0.1), math.exp(-0.1))                # a fall at 1 s

    def test_a_time_constant_of_0_passes_the_value_through(self) -> None:
        ema = EmaFilter(0.0)
        ema(0.0, 0.0)
        self.assertEqual(ema(5.0, 0.01), 5.0)
        self.assertEqual(ema(-2.0, 0.02), -2.0)

    def test_the_first_sample_is_the_starting_value(self) -> None:
        ema = EmaFilter(1.0)
        self.assertTrue(math.isnan(ema.value))
        self.assertEqual(ema(3.0, 10.0), 3.0)

    def test_the_rpm_smooths_as_with_its_old_rate_per_second(self) -> None:
        """White space's playhead: the old filter took `1 - (1 - alpha)^dt` of the way each sample; the new one has
        the time constant -1 / ln(1 - alpha)."""
        def time_constant(rate: float) -> float:
            return -1.0 / math.log(1.0 - rate) if rate < 1.0 else 0.0

        timestamps: list[float] = [0.0, 0.033, 0.05, 0.1, 0.133, 0.2, 0.25, 0.4, 0.41, 0.5]
        values: list[float] = [60.0, 61.5, 59.0, 62.0, 60.5, 58.0, 63.0, 60.0, 61.0, 59.5]
        for alpha in (0.03, 0.2, 0.5, 0.97, 1.0):
            old: float = values[0]
            for i in range(1, len(values)):
                dt: float = timestamps[i] - timestamps[i - 1]
                old += (values[i] - old) * (1.0 - pow(1.0 - alpha, dt))
            ema = EmaFilter(time_constant(alpha))
            ema.reset(values[0])
            new: float = run(ema, values, timestamps)
            self.assertAlmostEqual(new, old, places=9, msg=f'alpha {alpha}')


class OneEuroFilterTest(unittest.TestCase):
    def test_a_constant_input_holds(self) -> None:
        f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
        value: float = run(f, [2.5] * 100, [i / 60.0 for i in range(100)])
        self.assertEqual(value, 2.5)

    def test_it_follows_a_ramp(self) -> None:
        slow = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        fast = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        timestamps: list[float] = [i / 60.0 for i in range(300)]
        ramp: list[float] = [2.0 * t for t in timestamps]      # 2 per second, for 5 s
        slow_value: float = run(slow, ramp, timestamps)
        fast_value: float = run(fast, ramp, timestamps)
        # a first-order lowpass lags a ramp by its slope times its time constant, 1 / (2 pi cutoff)
        self.assertAlmostEqual(ramp[-1] - slow_value, 2.0 / (2.0 * math.pi), delta=0.001)
        self.assertLess(ramp[-1] - fast_value, (ramp[-1] - slow_value) / 2.0)   # the speed raises the cutoff


class AngleFilterTest(unittest.TestCase):
    def test_an_angle_across_pi_takes_the_short_way(self) -> None:
        f = AngleFilter(EmaFilter(0.1))
        f(3.0, 0.0)
        turn: float = 2.0 * math.pi - 6.0                       # from 3 to -3 the short way, through pi
        for i in range(1, 20):
            value: float = f(-3.0, i * 0.01)
            moved: float = (value - 3.0 + math.pi) % math.tau - math.pi
            self.assertGreater(moved, 0.0)                      # a positive turn, never back through 0
            self.assertLessEqual(moved, turn + 1e-12)
            self.assertTrue(-math.pi <= value < math.pi)
        self.assertAlmostEqual(f.value, (3.0 + turn * (1.0 - math.exp(-1.9)) + math.pi) % math.tau - math.pi)

    def test_an_ema_with_a_fall_time_follows_a_negative_turn_at_it(self) -> None:
        f = AngleFilter(EmaFilter(10.0, fall_time_constant=0.5))
        f(0.5, 0.0)
        self.assertAlmostEqual(f(-0.5, 0.5), 0.5 - 1.0 * (1.0 - math.exp(-1.0)))     # the fall time
        f.reset(-0.5)
        f(-0.5, 0.0)
        self.assertAlmostEqual(f(0.5, 0.5), -0.5 + 1.0 * (1.0 - math.exp(-0.05)))    # a positive turn: the rise time


if __name__ == '__main__':
    unittest.main()
