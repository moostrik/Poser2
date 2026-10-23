"""Tests for the rate limiters: a constant-rate cap, per second and rise/fall apart, on how fast a
feature may move (wired on the angles at white_space's LERP stage). The capping tests drive the
node's limiter with explicit times, since the node times itself by the wall clock; the node-level
tests cover the frame contract and the live settings."""

import unittest

import numpy as np

from modules.pose.features import AngleLandmark, Angles, AngleSymmetry
from modules.pose.nodes import AngleRateLimiter, AngleSymRateLimiter, RateLimiterSettings

from ._builders import frame, scalar

A = AngleLandmark


def _settings(max_increase: float, max_decrease: float) -> RateLimiterSettings:
    s = RateLimiterSettings()
    s.max_increase = max_increase
    s.max_decrease = max_decrease
    return s


def _vector(value: float) -> np.ndarray:
    return np.full(AngleSymmetry.length(), value, dtype=np.float64)


class RateLimitTest(unittest.TestCase):
    """The limiter itself, through a plain-vector node's instance, with explicit times."""

    def _limiter(self, max_increase: float = 2.0, max_decrease: float = 1.0):
        return AngleSymRateLimiter(_settings(max_increase, max_decrease))._limiter

    def test_the_first_value_initializes_at_once(self) -> None:
        limiter = self._limiter()
        limiter.update(_vector(0.5), current_time=0.0)
        np.testing.assert_allclose(limiter.value, _vector(0.5))

    def test_a_rise_is_capped_at_max_increase_per_second(self) -> None:
        limiter = self._limiter(max_increase=2.0)
        limiter.update(_vector(0.0), current_time=0.0)
        limiter.update(_vector(1.0), current_time=0.1)
        np.testing.assert_allclose(limiter.value, _vector(0.2), atol=1e-9)

    def test_a_fall_is_capped_at_max_decrease_per_second(self) -> None:
        limiter = self._limiter(max_decrease=1.0)
        limiter.update(_vector(0.5), current_time=0.0)
        limiter.update(_vector(-1.0), current_time=0.1)
        np.testing.assert_allclose(limiter.value, _vector(0.4), atol=1e-9)

    def test_arrives_then_holds(self) -> None:
        limiter = self._limiter(max_increase=2.0)
        limiter.update(_vector(0.0), current_time=0.0)
        for i in range(1, 11):
            limiter.update(_vector(0.6), current_time=0.1 * i)
        np.testing.assert_allclose(limiter.value, _vector(0.6), atol=1e-9)

    def test_nan_transitions_snap(self) -> None:
        limiter = self._limiter(max_increase=0.1, max_decrease=0.1)   # tight: a glide would show
        limiter.update(_vector(0.9), current_time=0.0)
        limiter.update(_vector(np.nan), current_time=0.1)             # gone: NaN at once
        self.assertTrue(np.all(np.isnan(limiter.value)))
        limiter.update(_vector(-0.9), current_time=0.2)               # back: the value at once
        np.testing.assert_allclose(limiter.value, _vector(-0.9))

    def test_the_clamp_range_bounds_the_value(self) -> None:
        limiter = self._limiter(max_increase=1000.0)
        limiter.update(_vector(0.0), current_time=0.0)
        limiter.update(_vector(5.0), current_time=1.0)
        np.testing.assert_allclose(limiter.value, _vector(1.0))       # AngleSymmetry's range top

    def test_reset_forgets_the_state(self) -> None:
        limiter = self._limiter()
        limiter.update(_vector(0.5), current_time=0.0)
        limiter.reset()
        self.assertTrue(np.all(np.isnan(limiter.value)))
        limiter.update(_vector(-0.5), current_time=10.0)              # a fresh start: at once
        np.testing.assert_allclose(limiter.value, _vector(-0.5))


class AngleRateLimitTest(unittest.TestCase):
    """The angles' limiter: the cap along the shortest arc, the output wrapped to [-π, π]."""

    def _limiter(self, max_increase: float, max_decrease: float):
        return AngleRateLimiter(_settings(max_increase, max_decrease))._limiter

    @staticmethod
    def _angles(value: float) -> np.ndarray:
        return np.full(Angles.length(), value, dtype=np.float64)

    def test_the_shortest_path_crosses_the_wrap(self) -> None:
        limiter = self._limiter(100.0, 100.0)
        limiter.update(self._angles(3.0), current_time=0.0)
        limiter.update(self._angles(-3.0), current_time=0.1)          # ~0.28 rad through π, not ~6 through 0
        np.testing.assert_allclose(limiter.value, self._angles(-3.0), atol=1e-6)

    def test_a_capped_move_heads_through_pi_not_through_zero(self) -> None:
        limiter = self._limiter(1.0, 1.0)
        limiter.update(self._angles(3.0), current_time=0.0)
        limiter.update(self._angles(-3.0), current_time=0.1)
        np.testing.assert_allclose(limiter.value, self._angles(3.1), atol=1e-6)


class FeatureRateLimiterTest(unittest.TestCase):
    """The node: the frame contract and the live settings."""

    def test_the_first_frame_passes_through_and_keeps_the_scores(self) -> None:
        node = AngleRateLimiter(_settings(1.0, 1.0))
        src = frame(features={Angles: scalar(Angles, {A.left_elbow: 1.2, A.right_elbow: -0.4}, score=0.8)})
        out = node.process(src)
        np.testing.assert_allclose(out[Angles].values[A.left_elbow], 1.2, atol=1e-6)
        self.assertAlmostEqual(float(out[Angles].get_score(A.left_elbow)), 0.8, places=6)

    def test_a_jump_is_held_back(self) -> None:
        node = AngleRateLimiter(_settings(1.0, 1.0))
        node.process(frame(t=0.0, features={Angles: scalar(Angles, {A.left_elbow: 0.0})}))
        out = node.process(frame(t=0.1, features={Angles: scalar(Angles, {A.left_elbow: 3.0})}))
        # The node times itself by the wall clock: the two calls are far less than a second apart,
        # so at 1 rad/s next to nothing of the 3 rad jump comes through.
        self.assertLess(float(out[Angles].values[A.left_elbow]), 1.0)

    def test_a_settings_change_takes_effect_live(self) -> None:
        settings = _settings(1.0, 1.0)
        node = AngleRateLimiter(settings)
        settings.max_increase = 5.0                                   # the panel's path: bind_all
        self.assertEqual(node._limiter.max_increase, 5.0)


if __name__ == "__main__":
    unittest.main()
