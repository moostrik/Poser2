"""The kinetic bridge's arithmetic: the trail weight and the numpy reference of its shaders."""

import math
import unittest

import numpy as np

from modules.flow import dead_zone, decay_weight, depth_speed, drop, fill_reservoir, hold_release, project, smooth_depth, stop_gate


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

    def test_stopping_toward_the_camera_is_a_release(self) -> None:
        fast, slow = np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 2.0]])
        acceleration, _ = project(fast, slow)
        self.assertAlmostEqual(acceleration[0], -2.0)

    def test_starting_toward_the_camera_is_an_onset(self) -> None:
        fast, slow = np.array([[0.0, 0.0, 2.0]]), np.array([[0.0, 0.0, 1.0]])
        acceleration, _ = project(fast, slow)
        self.assertAlmostEqual(acceleration[0], 1.0)

    def test_without_z_the_projection_is_the_2d_one(self) -> None:
        fast, slow = np.array([[1.0, 0.5]]), np.array([[0.2, 0.7]])
        flat = project(fast, slow)
        lifted = project(np.array([[1.0, 0.5, 0.0]]), np.array([[0.2, 0.7, 0.0]]))
        np.testing.assert_allclose(flat, lifted)

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


class ReleaseTest(unittest.TestCase):
    def test_onset_holds_nothing(self) -> None:
        self.assertEqual(hold_release(np.array([2.0]), np.zeros(1), 0.5)[0], 0.0)

    def test_one_stop_is_one_blob(self) -> None:
        release: list[float] = [0.0, 1.0, 0.1, 0.8, 0.1, 0.0]           # a flickering stop
        held = np.zeros(1)
        output: list[float] = []
        for value in release:
            held = hold_release(np.array([-value]), held, decay_weight(1 / 60, 100.0))
            output.append(float(held[0]))
        self.assertTrue(all(value > 0.6 for value in output[1:5]))      # it never falls away between the peaks
        self.assertGreater(output[1], output[-1])

    def test_a_full_stop_leaves_the_slow_trail(self) -> None:
        fast, slow = np.zeros((1, 2)), np.array([[1.0, 0.0]])
        velocity = drop(fast, slow, np.array([-1.0]), 1.0)
        np.testing.assert_allclose(velocity, slow)

    def test_no_drop_keeps_the_fast_trail(self) -> None:
        fast, slow = np.array([[0.5, 0.0]]), np.array([[1.0, 0.0]])
        np.testing.assert_allclose(drop(fast, slow, np.array([-0.5]), 0.0), fast)

    def test_an_onset_keeps_the_fast_trail(self) -> None:
        fast, slow = np.array([[2.0, 0.0]]), np.array([[1.0, 0.0]])
        np.testing.assert_allclose(drop(fast, slow, np.array([1.0]), 1.0), fast)

    def test_no_slow_trail_divides_by_nothing(self) -> None:
        velocity = drop(np.zeros((1, 2)), np.zeros((1, 2)), np.array([0.0]), 1.0)
        np.testing.assert_allclose(velocity, np.zeros((1, 2)))


class StopGateTest(unittest.TestCase):
    def test_a_steady_patch_lets_nothing_through(self) -> None:
        self.assertEqual(stop_gate(np.array([1.0]), np.array([1.0]), 1.0)[0], 0.0)

    def test_a_stopped_patch_lets_all_through(self) -> None:
        self.assertEqual(stop_gate(np.array([0.0]), np.array([1.0]), 1.0)[0], 1.0)

    def test_a_speeding_patch_lets_nothing_through(self) -> None:
        self.assertEqual(stop_gate(np.array([2.0]), np.array([1.0]), 1.0)[0], 0.0)

    def test_strength_0_is_no_gate(self) -> None:
        self.assertEqual(stop_gate(np.array([1.0]), np.array([1.0]), 0.0)[0], 1.0)

    def test_no_motion_divides_by_nothing(self) -> None:
        self.assertEqual(stop_gate(np.array([0.0]), np.array([0.0]), 1.0)[0], 0.0)

    def test_the_gate_scales_a_new_release(self) -> None:
        self.assertAlmostEqual(hold_release(np.array([-2.0]), np.zeros(1), 0.5, 0.25)[0], 0.5)


class DepthTest(unittest.TestCase):
    def test_the_dead_zone_clears_the_floor_and_stays_continuous(self) -> None:
        np.testing.assert_allclose(dead_zone(np.array([0.05, -0.05, 0.3, -0.3]), 0.1), [0.0, 0.0, 0.2, -0.2])

    def test_coming_closer_is_positive(self) -> None:
        self.assertAlmostEqual(depth_speed(np.array([1.9]), np.array([2.0]), 0.1, 0.5)[0], 1.0)

    def test_going_away_is_negative(self) -> None:
        self.assertAlmostEqual(depth_speed(np.array([2.1]), np.array([2.0]), 0.1, 0.5)[0], -1.0)

    def test_a_silhouette_edge_is_no_motion(self) -> None:
        self.assertEqual(depth_speed(np.array([1.2]), np.array([3.0]), 0.1, 0.1)[0], 0.0)

    def test_no_depth_at_either_frame_is_no_motion(self) -> None:
        self.assertEqual(depth_speed(np.array([0.0]), np.array([2.0]), 0.1, 5.0)[0], 0.0)
        self.assertEqual(depth_speed(np.array([2.0]), np.array([0.0]), 0.1, 5.0)[0], 0.0)

    def test_smoothing_halves_a_step_at_half_weight(self) -> None:
        self.assertAlmostEqual(smooth_depth(np.array([2.1]), np.array([2.0]), 0.5, 0.5)[0], 2.05)

    def test_a_jump_restarts_and_then_reads_as_no_motion(self) -> None:
        previous = np.array([3.0])
        smoothed = smooth_depth(np.array([1.2]), previous, 0.9, 0.1)
        self.assertAlmostEqual(smoothed[0], 1.2)
        self.assertEqual(depth_speed(smoothed, previous, 0.1, 0.1)[0], 0.0)

    def test_noise_averages_out_and_a_real_approach_survives(self) -> None:
        rng = np.random.default_rng(0)
        weight: float = decay_weight(1 / 60, 60.0)
        for approach, expected in ((0.0, 0.0), (1.0, 1.0)):                 # m/s
            depth = 2.0 - approach * np.arange(90) / 60
            noisy = depth + rng.normal(0.0, 0.012, 90)
            smoothed, speeds = np.array([noisy[0]]), []
            for value in noisy[1:]:
                last = smoothed
                smoothed = smooth_depth(np.array([value]), last, weight, 0.1)
                speeds.append(depth_speed(smoothed, last, 1 / 60, 0.1)[0])
            self.assertAlmostEqual(float(np.mean(speeds[30:])), expected, delta=0.1)


class ReservoirTest(unittest.TestCase):
    def test_motion_fills_it_by_speed_times_time(self) -> None:
        self.assertAlmostEqual(fill_reservoir(np.zeros(1), np.array([2.0]), np.zeros(1), 1.0, 0.5, 1.0)[0], 1.0)

    def test_it_leaks_without_motion(self) -> None:
        self.assertAlmostEqual(fill_reservoir(np.array([1.0]), np.zeros(1), np.zeros(1), 0.5, 0.1, 1.0)[0], 0.5)

    def test_a_release_drains_what_it_emits(self) -> None:
        drained = fill_reservoir(np.array([1.0]), np.zeros(1), np.array([2.0]), 1.0, 0.1, 1.0)
        self.assertAlmostEqual(drained[0], 0.8)

    def test_it_never_goes_below_zero(self) -> None:
        self.assertEqual(fill_reservoir(np.array([1.0]), np.zeros(1), np.array([100.0]), 1.0, 0.1, 1.0)[0], 0.0)


if __name__ == '__main__':
    unittest.main()
