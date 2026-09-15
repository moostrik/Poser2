"""Tests for WindowSimilarity's pairwise computation on windows built the way the app builds them (WindowNode)."""

import math
import unittest

import numpy as np

from modules.pose.analytics import WindowSimilarity, WindowSimilaritySettings
from modules.pose.features import AngleLandmark, Angles, AngleVelocity, LeaderScore, Similarity
from modules.pose.frame import FeatureWindow
from modules.pose.window import WindowNode, WindowNodeSettings

from ._builders import NUM_POSES, frame

T = 10
STEP = 0.05          # rad per frame on every joint
F = len(AngleLandmark)


def _window(per_frame: list[np.ndarray], feature: type = Angles, size: int = T) -> FeatureWindow:
    """Feed per-frame arrays (NaN = missing joint) through a WindowNode and return its last window."""
    settings = WindowNodeSettings()
    settings.window_size = size
    node = WindowNode(feature, settings)
    window = None
    for values in per_frame:
        v = values.astype(np.float32)
        s = np.where(np.isnan(v), 0.0, 1.0).astype(np.float32)
        window = node.process(frame(features={feature: feature(v, s)}))
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

    def test_missing_joint_is_not_compared_as_zero(self) -> None:
        # WindowNode stores a missing joint as 0.0 and marks it only in the mask; "joint missing" and
        # "joint measured at 0 rad" must not give the same similarity.
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

    def test_missing_joint_is_skipped_and_penalised_by_coverage(self) -> None:
        missing = _ramp()
        for values in missing:
            values[AngleLandmark.left_knee] = np.nan
        sim, lead = _similarity()._process({0: _window(_ramp()), 1: _window(missing)})
        for i, j in ((0, 1), (1, 0)):
            with self.subTest(pair=(i, j)):
                self.assertAlmostEqual(sim[i][j], (F - 1) / F, places=5)
                self.assertAlmostEqual(sim[i].get_score(j), (F - 1) / F, places=5)
                self.assertAlmostEqual(lead[i][j], 0.0, places=5)

    def test_unfilled_window_slots_are_not_frames(self) -> None:
        # Track 1 has only 3 frames, all far from track 0's pose. Its 7 unfilled slots hold 0.0 in the buffer;
        # if they counted as frames they would match track 0's 0-rad pose perfectly.
        still = [np.zeros(F) for _ in range(T)]
        new_arrival = [np.full(F, 1.5) for _ in range(3)]
        sim, lead = _similarity(use_time_penalty=False)._process({0: _window(still), 1: _window(new_arrival)})
        self.assertLess(sim[0][1], 0.1)
        self.assertLessEqual(lead[0][1], 2 / (T - 1) + 1e-6)

    def test_partial_window_still_matches_an_identical_track(self) -> None:
        few = _ramp()[:3]
        sim, lead = _similarity()._process({0: _window(few), 1: _window(few)})
        self.assertAlmostEqual(sim[0][1], 1.0, places=5)
        self.assertAlmostEqual(sim[0].get_score(1), 1.0, places=5)
        self.assertAlmostEqual(lead[0][1], 0.0, places=5)

    def test_track_without_current_angles_has_no_similarity(self) -> None:
        # Track 0's current frame has no angles at all: nothing to compare from its side. This must not abort
        # the batch (an all-NaN argmax raises).
        gone = _ramp()
        gone[-1] = np.full(F, np.nan)
        sim, lead = _similarity()._process({0: _window(gone), 1: _window(_ramp())})
        self.assertTrue(math.isnan(sim[0][1]))
        self.assertEqual(sim[0].get_score(1), 0.0)
        self.assertEqual(lead[0].get_score(1), 0.0)
        self.assertEqual(lead[0][1], 0.0)
        self.assertFalse(math.isnan(sim[1][0]))        # track 1's current pose still finds track 0's past frames

    def test_single_frame_window_compares_current_postures_only(self) -> None:
        # window_length 1 is posture similarity: only the current frames meet. Track 1's current pose equals
        # track 0's oldest, so a full window would score 1.0; here only the current poses, STEP*(T-1) apart, count.
        ws = _similarity(window_length=1, use_motion_weighting=False, use_velocity_similarity=False,
                         remap_low=0.0, remap_high=1.0)
        sim, lead = ws._process({0: _window(_ramp()), 1: _window(_ramp(lag=T - 1))})
        expected = math.exp(-((STEP * (T - 1)) / ws._config.angle_scale) ** 2)
        self.assertAlmostEqual(sim[1][0], expected, places=5)
        self.assertAlmostEqual(sim[0][1], expected, places=5)
        self.assertEqual(lead[1][0], 0.0)

        # Different histories, same current posture: full similarity.
        still = [np.full(F, STEP * (T - 1)) for _ in range(T)]
        sim, lead = ws._process({0: _window(_ramp()), 1: _window(still)})
        self.assertAlmostEqual(sim[0][1], 1.0, places=5)
        self.assertAlmostEqual(sim[1][0], 1.0, places=5)
        self.assertEqual(lead[0][1], 0.0)

    def test_missing_velocity_does_not_reduce_coverage(self) -> None:
        velocity = [np.full(F, STEP * 30.0) for _ in range(T)]
        gappy = [v.copy() for v in velocity]
        for values in gappy:
            values[AngleLandmark.head] = np.nan
        sim, _ = _similarity(use_velocity_similarity=True)._process(
            {0: _window(_ramp()), 1: _window(_ramp())},
            velocity_windows={0: _window(velocity, AngleVelocity), 1: _window(gappy, AngleVelocity)},
        )
        self.assertAlmostEqual(sim[0][1], 1.0, places=5)
        self.assertAlmostEqual(sim[0].get_score(1), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
