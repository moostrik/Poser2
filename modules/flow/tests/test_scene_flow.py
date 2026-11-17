"""The trail bridge's shaping of a velocity."""

import unittest

import numpy as np

from modules.flow import velocity_trail, velocity_change, step_peak, moving_mean, start_region_share, stop_region_share, change_velocity, \
                         decay_weight


class VelocityTrailTest(unittest.TestCase):
    def test_a_rising_speed_follows_the_attack_on_every_axis(self) -> None:
        stepped = velocity_trail(np.zeros((1, 3)), np.array([[1.0, 0.0, 1.0]]), 0.1, 0.9)
        np.testing.assert_allclose(stepped, [[0.9, 0.0, 0.9]])

    def test_a_rise_toward_the_camera_alone_follows_the_attack(self) -> None:
        stepped = velocity_trail(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), 0.1, 0.9)
        np.testing.assert_allclose(stepped, [[0.0, 0.0, 0.9]])

    def test_a_falling_xy_follows_the_release(self) -> None:
        stepped = velocity_trail(np.array([[1.0, 0.0, 0.0]]), np.zeros((1, 3)), 0.1, 0.9)
        np.testing.assert_allclose(stepped, [[0.9, 0.0, 0.0]])


class VelocityChangeTest(unittest.TestCase):
    STILL = np.zeros((1, 3))

    def test_a_speeding_up_motion_is_a_start_not_a_stop(self) -> None:
        fast, slow = np.array([[1.0, 0.0, 0.0]]), np.array([[0.25, 0.0, 0.0]])
        np.testing.assert_allclose(velocity_change(fast, slow, 1.0), [0.75])
        np.testing.assert_allclose(velocity_change(fast, slow, -1.0), [0.0])

    def test_a_slowing_motion_is_a_stop_along_the_motion_it_ended(self) -> None:
        fast, slow = self.STILL, np.array([[0.0, 0.5, 0.0]])
        stop = velocity_change(fast, slow, -1.0)
        np.testing.assert_allclose(stop, [0.5])
        np.testing.assert_allclose(change_velocity(slow, stop), [[0.0, 0.5, 0.0]])

    def test_a_steady_motion_has_no_change(self) -> None:
        moving = np.array([[0.3, -0.4, 0.0]])
        for sign in (1.0, -1.0):
            np.testing.assert_allclose(velocity_change(moving, moving, sign), [0.0])

    def test_the_step_peak(self) -> None:
        self.assertEqual(step_peak(0.0, 150.0), 1.0)
        self.assertAlmostEqual(step_peak(25.0, 150.0), 0.582, places=3)
        self.assertEqual(step_peak(150.0, 150.0), 0.0)
        self.assertEqual(step_peak(200.0, 150.0), 0.0)

    def test_a_sudden_stop_at_speed_v_reads_v(self) -> None:
        # The two trails stepped at 240 Hz through a stop from 0.4 widths/s to rest; the scaled fall's peak is the speed
        dt, v = 1.0 / 240.0, np.array([[0.4, 0.0, 0.0]])
        fast, slow, peak = v.copy(), v.copy(), 0.0
        fast_weight, slow_weight = decay_weight(dt, 25.0), decay_weight(dt, 150.0)
        for _ in range(240):
            fast, slow = fast_weight * fast, slow_weight * slow
            peak = max(peak, float(velocity_change(fast, slow, -1.0, scale=1.0 / step_peak(25.0, 150.0))[0]))
        self.assertAlmostEqual(peak, 0.4, delta=0.01)

    def test_the_region_scales_a_stop(self) -> None:
        np.testing.assert_allclose(velocity_change(self.STILL, np.array([[1.0, 0.0, 0.0]]), -1.0, 0.25), [0.25])

    def test_the_region_and_the_edge_split_the_whole_change(self) -> None:
        fast, slow = np.array([[1.0, 0.0, 0.0]]), np.array([[0.25, 0.0, 0.0]])
        start = velocity_change(fast, slow, 1.0, 0.4)
        bow = velocity_change(fast, slow, 1.0, 0.4, edge=True)
        np.testing.assert_allclose(start + bow, velocity_change(fast, slow, 1.0))

    def test_region_weight_0_is_all_start_and_stop_no_bow_or_wake(self) -> None:
        fast, slow = np.array([[1.0, 0.0, 0.0]]), np.array([[0.25, 0.0, 0.0]])
        share = start_region_share(np.array([0.3, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]), 0.0)
        np.testing.assert_allclose(velocity_change(fast, slow, 1.0, share, edge=True), [0.0])
        share = stop_region_share(self.region((0.3, 0.0, 0.0)), self.region((0.3, 0.0, 0.0)), 0.0)
        np.testing.assert_allclose(velocity_change(slow, fast, -1.0, share, edge=True), [0.0])

    @staticmethod
    def region(*xyz: tuple[float, float, float]) -> np.ndarray:
        """A region's pixels (N, 3) with these velocities."""
        return np.array(xyz, dtype=float)

    @staticmethod
    def start_shares(frames: list[np.ndarray], dt: float = 1.0 / 60.0) -> list[float]:
        """The share of a rise its region shares, per frame of the region's pixels: its mean moving velocity
        through the fast (25 ms) and the slow (150 ms) trail, as VelocityBandpass smooths it."""
        fast, slow, shares = np.zeros(3), np.zeros(3), []
        for pixels in frames:
            mean = moving_mean(pixels)
            fast_weight, slow_weight = decay_weight(dt, 25.0), decay_weight(dt, 150.0)
            fast = fast_weight * fast + (1.0 - fast_weight) * mean
            slow = slow_weight * slow + (1.0 - slow_weight) * mean
            shares.append(start_region_share(fast, slow, 1.0))
        return shares

    def test_jitter_on_a_still_region_is_no_start(self) -> None:
        # Every pixel moves, in every direction and in z: its mean is still nothing
        jitter = self.region((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.2, 0.3), (0.0, -0.2, -0.3))
        frames = [jitter if k % 2 else -jitter for k in range(30)]
        self.assertAlmostEqual(max(self.start_shares(frames)), 0.0)

    def test_a_limb_arriving_in_a_region_at_its_speed_is_a_bow_not_a_start(self) -> None:
        v = (0.3, 0.0, 0.1)
        frames = [self.region(v, *[(0.0, 0.0, 0.0)] * 7)] * 60                                  # in the region, settled
        frames += [self.region(*[v] * k, *[(0.0, 0.0, 0.0)] * (8 - k)) for k in range(2, 9)]     # more of it arrives
        start_share = max(self.start_shares(frames)[60:])
        self.assertLess(start_share, 0.01)                 # what the slow trail has left of settling
        self.assertGreater(1.0 - start_share, 0.99)       # the rest of the rise is the bow

    def test_a_region_speeding_up_is_a_start(self) -> None:
        frames = [self.region(*[(0.1, 0.0, 0.0)] * 8)] * 60
        frames += [self.region(*[(0.1 + 0.05 * k, 0.0, 0.0)] * 8) for k in range(1, 8)]
        self.assertGreater(max(self.start_shares(frames)[60:]), 0.3)

    def test_a_start_from_rest_is_a_start(self) -> None:
        frames = [self.region(*[(0.0, 0.0, 0.0)] * 8)] * 30 + [self.region(*[(0.0, 0.0, 0.3)] * 8)] * 5
        self.assertGreater(max(self.start_shares(frames)[30:]), 0.5)

    def test_a_stop_counts_as_far_as_its_region_slows_down(self) -> None:
        self.assertAlmostEqual(stop_region_share(self.region((0.5, 0.0, 0.0)), self.region((1.0, 0.0, 0.0)), 1.0), 0.5)
        self.assertAlmostEqual(stop_region_share(self.region((1.0, 0.0, 0.0)), self.region((0.5, 0.0, 0.0)), 1.0), 0.0)

    def test_a_turning_region_is_no_stop(self) -> None:
        # Its mean velocity shrinks as the directions spread; its mean speed does not
        fast = self.region((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
        slow = self.region((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(stop_region_share(fast, slow, 1.0), 0.0)

    def test_a_limb_leaving_a_region_is_mostly_a_wake(self) -> None:
        # Three of eight pixels it left: there the fast trail has let go and the slow one not yet; the region's mean
        # speed falls by a third of its share, so a third of the fall reads as a stop and the rest as the wake
        v = (0.3, 0.0, 0.0)
        fast = self.region(*[v] * 5, *[(0.0, 0.0, 0.0)] * 3)
        slow = self.region(*[v] * 5, *[(0.2, 0.0, 0.0)] * 3)
        share = stop_region_share(fast, slow, 1.0)
        self.assertLess(share, 0.5)
        self.assertGreater(1.0 - share, 0.5)

    def test_a_steady_region_lets_no_change_through_and_region_weight_0_lets_all(self) -> None:
        steady = self.region((0.7, 0.0, 0.0))
        self.assertAlmostEqual(stop_region_share(steady, steady, 1.0), 0.0)
        self.assertAlmostEqual(stop_region_share(steady, steady, 0.0), 1.0)
        self.assertAlmostEqual(start_region_share(steady[0], steady[0], 1.0), 0.0)
        self.assertAlmostEqual(start_region_share(steady[0], steady[0], 0.0), 1.0)


if __name__ == '__main__':
    unittest.main()
