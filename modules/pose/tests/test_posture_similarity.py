"""Tests for the posture kernel: two poses scored on their own, and the same numbers as WindowSimilarity at
window length 1 — the guard that the two callers cannot drift apart."""

import math
import unittest

import numpy as np

from modules.pose.analytics import WindowSimilarity, WindowSimilaritySettings, posture_similarity, joint_mask
from modules.pose.features import Angles, AngleLandmark
from modules.pose.window import WindowNode, WindowNodeSettings

from ._builders import NUM_POSES, frame

F = len(AngleLandmark)
TOLERANCE = 45.0        # °


def _angles(values: np.ndarray) -> Angles:
    v = np.asarray(values, dtype=np.float32)
    return Angles(v, np.where(np.isnan(v), 0.0, 1.0).astype(np.float32))


def _settings(**fields) -> WindowSimilaritySettings:
    cfg = WindowSimilaritySettings()
    cfg.max_poses = NUM_POSES
    cfg.window_length = 1
    cfg.angle_tolerance = TOLERANCE
    cfg.use_velocity_similarity = False
    cfg.use_motion_weighting = False
    cfg.use_time_penalty = False
    cfg.remap_low = 0.0
    cfg.remap_high = 1.0
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def _window(angles: Angles):
    settings = WindowNodeSettings()
    settings.window_size = 1
    return WindowNode(Angles, settings).process(frame(features={Angles: angles}))


ARMS = (AngleLandmark.left_shoulder, AngleLandmark.right_shoulder, AngleLandmark.left_elbow, AngleLandmark.right_elbow)


def _arms_only(cfg: WindowSimilaritySettings) -> WindowSimilaritySettings:
    for landmark in AngleLandmark:
        setattr(cfg.joints, landmark.name, landmark in ARMS)
    return cfg


class PostureSimilarityTest(unittest.TestCase):
    def test_identical_poses_are_fully_alike(self) -> None:
        a = _angles(np.full(F, 0.3))
        self.assertAlmostEqual(posture_similarity(a, a, _settings()), 1.0, places=6)

    def test_one_joint_a_tolerance_off_reads_the_kernel(self) -> None:
        a = _angles(np.zeros(F))
        b_values = np.zeros(F)
        b_values[AngleLandmark.left_elbow] = math.radians(TOLERANCE)
        expected = F / ((F - 1) + 1.0 / math.exp(-1.0))        # harmonic mean of eight 1s and one 1/e
        self.assertAlmostEqual(posture_similarity(a, _angles(b_values), _settings()), expected, places=6)

    def test_a_missing_joint_is_skipped_and_scaled_by_coverage(self) -> None:
        a_values = np.zeros(F)
        a_values[AngleLandmark.head] = np.nan
        self.assertAlmostEqual(posture_similarity(_angles(a_values), _angles(np.zeros(F)), _settings()),
                               (F - 1) / F, places=6)

    def test_no_shared_joint_is_nan(self) -> None:
        a_values = np.full(F, np.nan)
        a_values[AngleLandmark.head] = 0.0
        b_values = np.full(F, np.nan)
        b_values[AngleLandmark.left_knee] = 0.0
        self.assertTrue(math.isnan(posture_similarity(_angles(a_values), _angles(b_values), _settings())))

    def test_the_remap_applies(self) -> None:
        a = _angles(np.zeros(F))
        b_values = np.zeros(F)
        b_values[AngleLandmark.left_elbow] = math.radians(TOLERANCE)
        raw = F / ((F - 1) + math.e)
        cfg = _settings(remap_low=0.2, remap_high=0.9)
        self.assertAlmostEqual(posture_similarity(a, _angles(b_values), cfg), (raw - 0.2) / 0.7, places=6)

    def test_equals_window_similarity_at_window_length_one(self) -> None:
        # Two poses that differ on several joints, with a missing joint and a non-trivial remap: the pair
        # function and the window module must agree to the last digit.
        a_values = np.linspace(-1.0, 1.0, F)
        b_values = a_values + np.linspace(0.0, 0.9, F)
        a_values[AngleLandmark.right_knee] = np.nan
        a, b = _angles(a_values), _angles(b_values)
        cfg = _settings(angle_tolerance=34.4, remap_low=0.2, remap_high=0.9)
        sim, _ = WindowSimilarity(cfg)._process({0: _window(a), 1: _window(b)})
        self.assertAlmostEqual(posture_similarity(a, b, cfg), sim[0][1], places=6)
        self.assertAlmostEqual(posture_similarity(b, a, cfg), sim[1][0], places=6)


class JointSelectionTest(unittest.TestCase):
    def test_joint_mask_follows_the_landmark_order(self) -> None:
        cfg = _arms_only(_settings())
        np.testing.assert_array_equal(joint_mask(cfg.joints), [True] * 4 + [False] * 5)
        np.testing.assert_array_equal(joint_mask(_settings().joints), [True] * F)

    def test_an_unselected_joint_that_differs_does_not_count(self) -> None:
        a = _angles(np.zeros(F))
        b_values = np.zeros(F)
        b_values[AngleLandmark.left_knee] = math.pi
        b_values[AngleLandmark.head] = math.pi
        self.assertAlmostEqual(posture_similarity(a, _angles(b_values), _arms_only(_settings())), 1.0, places=6)

    def test_coverage_is_over_the_selected_joints(self) -> None:
        a_values = np.zeros(F)
        a_values[AngleLandmark.right_elbow] = np.nan           # one arm joint missing
        a_values[AngleLandmark.left_knee] = np.nan             # a leg joint missing: not counted
        self.assertAlmostEqual(posture_similarity(_angles(a_values), _angles(np.zeros(F)), _arms_only(_settings())),
                               3 / 4, places=6)

    def test_no_selected_joint_is_nan(self) -> None:
        cfg = _settings()
        for landmark in AngleLandmark:
            setattr(cfg.joints, landmark.name, False)
        a = _angles(np.zeros(F))
        self.assertTrue(math.isnan(posture_similarity(a, a, cfg)))

    def test_equals_window_similarity_with_the_arms_selected(self) -> None:
        a_values = np.linspace(-1.0, 1.0, F)
        b_values = a_values + np.linspace(0.0, 0.9, F)
        a_values[AngleLandmark.right_elbow] = np.nan
        a, b = _angles(a_values), _angles(b_values)
        cfg = _arms_only(_settings(angle_tolerance=34.4, remap_low=0.2, remap_high=0.9))
        sim, _ = WindowSimilarity(cfg)._process({0: _window(a), 1: _window(b)})
        self.assertAlmostEqual(posture_similarity(a, b, cfg), sim[0][1], places=6)
        self.assertAlmostEqual(sim[0].scores[1], 3 / 4, places=6)


if __name__ == "__main__":
    unittest.main()
