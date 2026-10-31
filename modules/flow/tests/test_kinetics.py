"""The kinetic bridge's arithmetic: the trail weight and the numpy reference of its shaders."""

import math
import unittest

import numpy as np

from modules.flow import decay_weight, project


def trails(flow: list[float], fast_time: float = 25.0, slow_time: float = 150.0, dt: float = 1 / 60) -> tuple[np.ndarray, np.ndarray]:
    """The two trails over a 1D speed along +x, one value per frame, shape (frames, 2)."""
    fast_weight, slow_weight = decay_weight(dt, fast_time), decay_weight(dt, slow_time)
    fast, slow = 0.0, 0.0
    fasts, slows = [], []
    for speed in flow:
        fast = fast_weight * fast + (1.0 - fast_weight) * speed
        slow = slow_weight * slow + (1.0 - slow_weight) * speed
        fasts.append((fast, 0.0))
        slows.append((slow, 0.0))
    return np.array(fasts), np.array(slows)


class DecayWeightTest(unittest.TestCase):
    def test_one_time_constant_leaves_one_over_e(self) -> None:
        self.assertAlmostEqual(decay_weight(0.025, 25.0), 1 / math.e)

    def test_steps_compose_whatever_the_frame_rate(self) -> None:
        self.assertAlmostEqual(decay_weight(1 / 60, 150.0) ** 2, decay_weight(1 / 30, 150.0))

    def test_no_time_constant_forgets_at_once(self) -> None:
        self.assertEqual(decay_weight(1 / 60, 0.0), 0.0)


class ProjectTest(unittest.TestCase):
    def test_along_the_motion_is_acceleration(self) -> None:
        acceleration, turn = project(np.array([[2.0, 0.0]]), np.array([[1.0, 0.0]]))
        self.assertAlmostEqual(acceleration[0], 1.0)
        self.assertAlmostEqual(turn[0], 0.0)

    def test_slowing_down_is_negative(self) -> None:
        acceleration, _ = project(np.array([[0.0, 1.0]]), np.array([[0.0, 3.0]]))
        self.assertAlmostEqual(acceleration[0], -2.0)

    def test_across_the_motion_is_signed_turn(self) -> None:
        acceleration, turn = project(np.array([[1.0, 0.5]]), np.array([[1.0, 0.0]]))
        self.assertGreater(turn[0], 0.0)                                # left of +x
        self.assertGreater(acceleration[0], 0.0)
        _, turn_right = project(np.array([[1.0, -0.5]]), np.array([[1.0, 0.0]]))
        self.assertAlmostEqual(turn_right[0], -turn[0])

    def test_a_steady_motion_is_silent(self) -> None:
        fast, slow = trails([1.0] * 240)
        acceleration, turn = project(fast, slow)
        self.assertAlmostEqual(acceleration[-1], 0.0, places=4)
        self.assertAlmostEqual(turn[-1], 0.0, places=4)

    def test_no_motion_is_zero_without_dividing_by_it(self) -> None:
        acceleration, turn = project(np.zeros((1, 2)), np.zeros((1, 2)))
        self.assertEqual(acceleration[0], 0.0)
        self.assertEqual(turn[0], 0.0)

    def test_a_stopped_fast_trail_still_reads_the_release(self) -> None:
        acceleration, turn = project(np.zeros((1, 2)), np.array([[3.0, 4.0]]))
        self.assertAlmostEqual(acceleration[0], -5.0)
        self.assertAlmostEqual(turn[0], 0.0)

    def test_a_flick_is_onset_then_quiet_then_release(self) -> None:
        moving: int = 60
        fast, slow = trails([0.0] * 10 + [1.0] * moving + [0.0] * 60)
        acceleration, _ = project(fast, slow)
        self.assertGreater(acceleration[10:20].max(), 0.4)              # the start
        self.assertLess(np.abs(acceleration[10 + moving - 5:10 + moving]).max(), 0.05)  # mid-flick
        self.assertLess(acceleration[10 + moving:].min(), -0.4)         # the stop

    def test_a_slow_wave_stays_quiet(self) -> None:
        t = np.arange(600) / 60
        fast, slow = trails(list(1.0 + 0.5 * np.sin(2 * np.pi * 0.25 * t)))
        acceleration, _ = project(fast, slow)
        self.assertLess(np.abs(acceleration[300:]).max(), 0.1)


if __name__ == '__main__':
    unittest.main()
