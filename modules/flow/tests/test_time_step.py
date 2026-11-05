"""The fluid's time step: fixed, or the measured tick through an Ema."""

import unittest

from modules.flow import TimeStep, TimeStepMode


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

    def test_a_long_stall_steps_at_most_a_second(self) -> None:
        step = TimeStep(60.0)
        self.assertAlmostEqual(step.update(TimeStepMode.MEASURED, 5.0, 60.0, 0.0), 1.0)


if __name__ == '__main__':
    unittest.main()
