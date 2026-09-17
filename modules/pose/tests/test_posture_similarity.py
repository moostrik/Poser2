"""Tests for the posture module: the distance in degrees, the similarity of a distance, the pairwise rows."""

import math
import unittest

import numpy as np

from modules.pose.analytics import (PostureSimilarity, PostureSimilaritySettings, SimilarityResult,
                                    joint_distances, joint_mask, posture_distance, posture_similarity)
from modules.pose.features import Angles, AngleLandmark, Similarity

from ._builders import NUM_POSES, frame

F = len(AngleLandmark)
L = AngleLandmark
ARMS = (L.left_shoulder, L.right_shoulder, L.left_elbow, L.right_elbow)


def _angles(degrees: dict[AngleLandmark, float] | float = 0.0, missing: tuple[AngleLandmark, ...] = ()) -> Angles:
    """Every joint at 0°, or at ``degrees`` (one value for all, or per joint); ``missing`` joints are NaN."""
    values = np.zeros(F, dtype=np.float32)
    if isinstance(degrees, dict):
        for landmark, value in degrees.items():
            values[landmark] = math.radians(value)
    else:
        values[:] = math.radians(degrees)
    for landmark in missing:
        values[landmark] = np.nan
    return Angles(values, np.where(np.isnan(values), 0.0, 1.0).astype(np.float32))


def _settings(arms_only: bool = True, **fields) -> PostureSimilaritySettings:
    cfg = PostureSimilaritySettings()
    cfg.max_poses = NUM_POSES
    cfg.angle_tolerance = 20.0
    if arms_only:
        for landmark in AngleLandmark:
            setattr(cfg.joints, landmark.name, landmark in ARMS)
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


class JointDistancesTest(unittest.TestCase):
    def test_the_difference_wraps_the_short_way(self) -> None:
        d = joint_distances(_angles({L.left_shoulder: 170.0}), _angles({L.left_shoulder: -170.0}))
        self.assertAlmostEqual(d[L.left_shoulder], 20.0, places=3)

    def test_missing_and_unselected_joints_are_nan(self) -> None:
        mask = joint_mask(_settings().joints)
        d = joint_distances(_angles(missing=(L.left_elbow,)), _angles(30.0), mask)
        self.assertTrue(math.isnan(d[L.left_elbow]))          # missing on one side
        self.assertTrue(math.isnan(d[L.left_knee]))           # not selected
        self.assertAlmostEqual(d[L.right_elbow], 30.0, places=3)


class PostureDistanceTest(unittest.TestCase):
    def test_every_joint_the_same_distance_apart_is_that_distance(self) -> None:
        distance, coverage = posture_distance(_angles(0.0), _angles(19.0), _settings())
        self.assertAlmostEqual(distance, 19.0, places=3)
        self.assertEqual(coverage, 1.0)

    def test_one_odd_joint_is_forgiven_by_the_factor(self) -> None:
        # forgiveness 1.4: a lone joint at 28° with the other three exact reads 28 / 1.4 = 20.
        distance, _ = posture_distance(_angles(), _angles({L.left_elbow: 28.0}), _settings())
        self.assertAlmostEqual(distance, 20.0, places=3)

    def test_forgiveness_one_is_the_worst_joint_alone(self) -> None:
        distance, _ = posture_distance(_angles(), _angles({L.left_elbow: 28.0}), _settings(forgiveness=1.0))
        self.assertAlmostEqual(distance, 28.0, places=3)

    def test_below_one_point_one_is_none_and_never_overflows(self) -> None:
        distance, _ = posture_distance(_angles(), _angles(170.0), _settings(forgiveness=1.05))
        self.assertAlmostEqual(distance, 170.0, places=3)

    def test_the_sentence_holds_with_a_joint_missing(self) -> None:
        # Three joints compared: the exponent follows them, so the lone joint is still forgiven by 1.4.
        a = _angles(missing=(L.right_elbow,))
        distance, coverage = posture_distance(a, _angles({L.left_elbow: 28.0}), _settings())
        self.assertAlmostEqual(distance, 20.0, places=3)
        self.assertAlmostEqual(coverage, 0.75, places=6)

    def test_unselected_joints_do_not_count(self) -> None:
        distance, coverage = posture_distance(_angles(), _angles({L.left_knee: 90.0, L.head: 90.0}), _settings())
        self.assertAlmostEqual(distance, 0.0, places=6)
        self.assertEqual(coverage, 1.0)

    def test_no_shared_joint_is_nan(self) -> None:
        a = _angles(missing=ARMS)
        distance, coverage = posture_distance(a, _angles(), _settings())
        self.assertTrue(math.isnan(distance))
        self.assertEqual(coverage, 0.0)

    def test_one_joint_compared_is_that_joint(self) -> None:
        a = _angles(missing=(L.right_shoulder, L.left_elbow, L.right_elbow))
        distance, coverage = posture_distance(a, _angles({L.left_shoulder: 33.0}), _settings())
        self.assertAlmostEqual(distance, 33.0, places=3)
        self.assertAlmostEqual(coverage, 0.25, places=6)


class PostureSimilarityScoreTest(unittest.TestCase):
    def test_the_tolerance_is_the_slack_at_both_ends(self) -> None:
        cfg = _settings()
        self.assertEqual(posture_similarity(0.0, cfg), 1.0)
        self.assertEqual(posture_similarity(20.0, cfg), 1.0)          # exactly 1 on the plateau
        self.assertLess(posture_similarity(21.0, cfg), 1.0)
        self.assertAlmostEqual(posture_similarity(90.0, cfg), 0.5, places=6)
        self.assertEqual(posture_similarity(160.0, cfg), 0.0)
        self.assertEqual(posture_similarity(180.0, cfg), 0.0)
        self.assertTrue(math.isnan(posture_similarity(math.nan, cfg)))


class PostureSimilarityRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.out: list[SimilarityResult] = []
        self.node = PostureSimilarity(_settings())
        self.node.add_similarity_callback(self.out.append)

    def _rows(self, poses: dict[int, Angles]) -> dict[int, Similarity]:
        self.node.process({id: frame(track_id=id, features={Angles: a}) for id, a in poses.items()})
        return self.out[-1].similarity

    def test_rows_are_indexed_by_id_and_the_self_slot_is_empty(self) -> None:
        rows = self._rows({1: _angles(0.0), 3: _angles(10.0)})
        self.assertEqual(set(rows), {1, 3})
        self.assertEqual(rows[1][3], 1.0)
        self.assertEqual(rows[3][1], 1.0)
        self.assertEqual(rows[1].get_score(3), 1.0)
        for slot in (0, 1, 2):
            self.assertTrue(math.isnan(rows[1][slot]))
        self.assertEqual(self.out[-1].leader_score, {})

    def test_the_rows_carry_the_graded_similarity(self) -> None:
        rows = self._rows({0: _angles(0.0), 1: _angles(90.0)})
        self.assertAlmostEqual(rows[0][1], 0.5, places=5)
        self.assertAlmostEqual(rows[1][0], 0.5, places=5)

    def test_the_rows_match_the_pair_functions(self) -> None:
        a = _angles({L.left_shoulder: 40.0, L.left_elbow: 75.0}, missing=(L.right_elbow,))
        b = _angles({L.left_shoulder: 10.0, L.right_shoulder: 50.0})
        rows = self._rows({0: a, 1: b})
        distance, coverage = posture_distance(a, b, _settings())
        self.assertAlmostEqual(rows[0][1], posture_similarity(distance, _settings()), places=5)
        self.assertAlmostEqual(rows[0].get_score(1), coverage, places=6)

    def test_a_pose_alone_has_a_row_of_nothing(self) -> None:
        rows = self._rows({2: _angles(0.0)})
        self.assertTrue(np.isnan(rows[2].values).all())

    def test_a_pair_sharing_no_selected_joint_is_nan(self) -> None:
        rows = self._rows({0: _angles(missing=ARMS), 1: _angles(0.0)})
        self.assertTrue(math.isnan(rows[0][1]))
        self.assertEqual(rows[0].get_score(1), 0.0)


if __name__ == "__main__":
    unittest.main()
