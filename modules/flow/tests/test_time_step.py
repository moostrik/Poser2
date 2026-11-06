"""The fluid's time step: fixed, or the measured tick through an Ema."""

import math
import unittest

from modules.flow import MAX_STEP, TimeStep, TimeStepMode, hold_tick, viscosity_iterations


class TimeStepTest(unittest.TestCase):
    def test_fixed_ignores_the_render_rate(self) -> None:
        step = TimeStep()
        for dt in (1 / 30, 1 / 144, 1 / 7):
            self.assertAlmostEqual(step.update(TimeStepMode.FIXED, dt, 50.0, 1.0), 1 / 50)

    def test_no_smoothing_follows_each_frame(self) -> None:
        step = TimeStep()
        self.assertAlmostEqual(step.update(TimeStepMode.MEASURED, 1 / 30, 60.0, 0.0), 1 / 30)

    def test_a_stall_moves_the_smoothed_step_a_little(self) -> None:
        step = TimeStep(60.0)
        dt = step.update(TimeStepMode.MEASURED, 0.1, 60.0, 1.0)     # one 100 ms frame
        self.assertGreater(dt, 1 / 60)
        self.assertLess(dt, 1 / 40)                                  # a tenth of the way to 1/10

    def test_a_new_rate_is_reached_within_a_few_smoothing_times(self) -> None:
        step = TimeStep(60.0)
        for _ in range(30 * 4):                                      # 4 s at 30 fps, smoothing 1 s
            dt = step.update(TimeStepMode.MEASURED, 1 / 30, 60.0, 1.0)
        self.assertAlmostEqual(dt, 1 / 30, delta=0.001)



class SlowFrameTest(unittest.TestCase):
    def test_a_slow_frame_steps_at_most_the_longest_step(self) -> None:
        self.assertEqual(TimeStep().update(TimeStepMode.MEASURED, 0.25, 60.0, 0.0), MAX_STEP)
        self.assertEqual(TimeStep().update(TimeStepMode.FIXED, 0.0, 4.0, 0.0), MAX_STEP)

    def test_viscosity_takes_fewer_iterations_on_a_short_step_and_never_more(self) -> None:
        self.assertEqual(viscosity_iterations(40, 1.0 / 60.0), 40)
        self.assertEqual(viscosity_iterations(40, 1.0 / 120.0), 20)
        self.assertEqual(viscosity_iterations(40, 0.25), 40)
        self.assertEqual(viscosity_iterations(40, 0.0), 1)


class HoldTickTest(unittest.TestCase):
    def test_the_first_tick_sets_the_mean(self) -> None:
        self.assertEqual(hold_tick(1 / 60, None), (1 / 60, 1 / 60))

    def test_a_stall_is_held_to_twice_the_mean(self) -> None:
        dt, mean = hold_tick(5.0, 1 / 60)
        self.assertAlmostEqual(dt, 2 / 60)
        self.assertGreater(mean, 1 / 60)
        self.assertLess(mean, 2 / 60)

    def test_a_clock_step_is_at_least_the_shortest_tick(self) -> None:
        self.assertAlmostEqual(hold_tick(0.0, 1 / 60)[0], 1 / 240)

    def test_a_tick_that_is_not_one_takes_the_mean(self) -> None:
        self.assertAlmostEqual(hold_tick(math.nan, 1 / 50)[0], 1 / 50)
        self.assertAlmostEqual(hold_tick(math.inf, None)[0], 1 / 240)


if __name__ == '__main__':
    unittest.main()
