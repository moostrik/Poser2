"""The explosion guards without GL: the guard's report from its counts, and the velocity's soft top speed."""

import unittest

import numpy as np

from modules.flow import guard_report, soft_limit


class TestGuardReport(unittest.TestCase):

    def test_quiet_when_every_count_is_zero(self) -> None:
        self.assertEqual(guard_report('fluid', ('velocity', 'density'), [0, 0], 1.0), [])

    def test_one_warning_per_field_with_bad_values(self) -> None:
        messages = guard_report('fluid', ('velocity', 'density', 'temperature'), [1240, 0, 3], 1.02)
        self.assertEqual(messages, ['fluid: 1240 non-finite velocity values set to 0 in the last 1.0 s',
                                    'fluid: 3 non-finite temperature values set to 0 in the last 1.0 s'])


class TestSoftLimit(unittest.TestCase):

    def test_off_at_zero(self) -> None:
        velocity = np.array([[100.0, 0.0, 0.0]])
        np.testing.assert_array_equal(soft_limit(velocity, 0.0), velocity)

    def test_untouched_up_to_half(self) -> None:
        velocity = np.array([[1.0, 0.5, -0.5], [0.0, 1.5, 0.0]])
        np.testing.assert_allclose(soft_limit(velocity, 3.0), velocity)

    def test_never_past_the_top_and_direction_kept(self) -> None:
        velocity = np.array([[3.0, 0.0, 0.0], [0.0, -10.0, 0.0], [1e4, 1e4, 0.0]])
        limited = soft_limit(velocity, 3.0)
        speed = np.linalg.norm(limited, axis=-1)
        self.assertTrue(np.all(speed <= 3.0 + 1e-9))
        self.assertTrue(np.all(speed > 1.5))
        np.testing.assert_allclose(limited / speed[:, None], velocity / np.linalg.norm(velocity, axis=-1)[:, None])

    def test_smooth_at_the_knee(self) -> None:
        below = soft_limit(np.array([[1.5 - 1e-6, 0.0, 0.0]]), 3.0)[0, 0]
        above = soft_limit(np.array([[1.5 + 1e-6, 0.0, 0.0]]), 3.0)[0, 0]
        self.assertAlmostEqual((above - below) / 2e-6, 1.0, places=4)

    def test_rises_with_the_speed(self) -> None:
        speeds = np.linspace(0.0, 20.0, 200)
        limited = soft_limit(np.stack([speeds, np.zeros_like(speeds), np.zeros_like(speeds)], axis=-1), 3.0)[:, 0]
        self.assertTrue(np.all(np.diff(limited) >= 0.0))


if __name__ == '__main__':
    unittest.main()
