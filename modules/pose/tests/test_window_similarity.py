"""Tests for WindowSimilarity's pairwise computation on windows built the way the app builds them (WindowNode)."""

import math
import unittest

import numpy as np

from modules.pose.analytics import WindowSimilarity, WindowSimilaritySettings
from modules.pose.features import AngleLandmark, Angles, LeaderScore, Similarity
from modules.pose.frame import FeatureWindow
from modules.pose.window import WindowNode, WindowNodeSettings

from ._builders import NUM_POSES, frame

T = 10
STEP = 0.05          # rad per frame on every joint


def _window(angles_per_frame: list[np.ndarray]) -> FeatureWindow:
    """Feed per-frame angle arrays (NaN = missing joint) through a WindowNode and return its last window."""
    settings = WindowNodeSettings()
    settings.window_size = len(angles_per_frame)
    node = WindowNode(Angles, settings)
    window = None
    for values in angles_per_frame:
        v = values.astype(np.float32)
        s = np.where(np.isnan(v), 0.0, 1.0).astype(np.float32)
        window = node.process(frame(features={Angles: Angles(v, s)}))
    return window


def _ramp(lag: int = 0, offset: float = 0.0) -> list[np.ndarray]:
    """Every joint follows offset + STEP * (t - lag)."""
    n = len(AngleLandmark)
    return [np.full(n, offset + STEP * (t - lag)) for t in range(T)]


def _similarity(**fields) -> WindowSimilarity:
    settings = WindowSimilaritySettings()
    settings.max_poses = NUM_POSES
    settings.window_length = T
    for name, value in fields.items():
        setattr(settings, name, value)
    return WindowSimilarity(settings)


class WindowSimilarityTest(unittest.TestCase):
    def test_fewer_than_two_tracks_gives_nothing(self) -> None:
        self.assertEqual(_similarity()._process({0: _window(_ramp())}), ({}, {}))

    def test_identical_movement_is_fully_similar_and_synchronised(self) -> None:
        sim, lead = _similarity()._process({0: _window(_ramp()), 1: _window(_ramp())})
        self.assertAlmostEqual(sim[0][1], 1.0, places=5)
        self.assertAlmostEqual(sim[1][0], 1.0, places=5)
        self.assertAlmostEqual(lead[0][1], 0.0, places=5)
        self.assertAlmostEqual(lead[1][0], 0.0, places=5)

    def test_outputs_are_contract_valid_features(self) -> None:
        sim, lead = _similarity()._process({0: _window(_ramp()), 1: _window(_ramp(offset=0.4))})
        for feature in (*sim.values(), *lead.values()):
            ok, err = feature.validate()
            self.assertTrue(ok, err)
        self.assertIsInstance(sim[0], Similarity)
        self.assertIsInstance(lead[0], LeaderScore)

    def test_different_poses_are_less_similar(self) -> None:
        sim, _ = _similarity()._process({0: _window(_ramp()), 1: _window(_ramp(offset=1.0))})
        self.assertLess(sim[0][1], 0.5)

    def test_leader_score_measures_the_lag(self) -> None:
        # Track 1 repeats track 0's movement k frames later: track 1's current pose is where track 0 was
        # k frames ago, so from track 1's view track 0 leads by k / (T - 1); from track 0's view nobody leads.
        k = 3
        _, lead = _similarity(use_time_penalty=False)._process({0: _window(_ramp()), 1: _window(_ramp(lag=k))})
        self.assertAlmostEqual(lead[1][0], k / (T - 1), places=5)
        self.assertAlmostEqual(lead[0][1], 0.0, places=5)

    def test_values_are_indexed_by_track_id_and_self_is_empty(self) -> None:
        sim, lead = _similarity()._process({1: _window(_ramp()), 3: _window(_ramp())})
        self.assertEqual(set(sim), {1, 3})
        row = sim[1]
        self.assertEqual(len(row), NUM_POSES)
        self.assertFalse(math.isnan(row[3]))
        for other in (0, 1, 2):
            self.assertTrue(math.isnan(row[other]), other)
            self.assertEqual(row.get_score(other), 0.0)
        self.assertEqual(lead[1].get_score(3), 1.0)
        self.assertEqual(lead[1].get_score(1), 0.0)

    @unittest.expectedFailure
    def test_missing_joint_is_not_compared_as_zero(self) -> None:
        # WindowNode stores a missing joint as 0.0 and marks it only in the mask; WindowSimilarity looks for
        # NaN in the values and never reads the mask, so "joint missing" and "joint measured at 0 rad" give
        # the same similarity.
        knee = AngleLandmark.left_knee
        a = _ramp()
        for values in a:
            values[knee] = 0.0
        measured = [v.copy() for v in a]
        missing = [v.copy() for v in a]
        for values in missing:
            values[knee] = np.nan

        ws = _similarity()
        sim_measured, _ = ws._process({0: _window(a), 1: _window(measured)})
        sim_missing, _ = ws._process({0: _window(a), 1: _window(missing)})
        same = (np.allclose(sim_measured[0].values, sim_missing[0].values, equal_nan=True)
                and np.allclose(sim_measured[0].scores, sim_missing[0].scores))
        self.assertFalse(same)


if __name__ == "__main__":
    unittest.main()
