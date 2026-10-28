"""Tests for the chase and lerp interpolator nodes and the InterpolatorPipeline that merges their outputs."""

import math
import unittest

import numpy as np

from modules.pose.features import AngleLandmark, Angles, AngleVelocity, Points2D, PointLandmark, Similarity
from modules.pose.frame import Frame
from modules.pose.nodes import (
    AngleChaseInterpolator, AngleLerpInterpolator, AngleVelLerpInterpolator, ChaseInterpolatorSettings,
    LerpInterpolatorSettings, PointChaseInterpolator, SimilarityChaseInterpolator,
)
from modules.pose.trackers import InterpolatorPipeline

from ._builders import frame, points, scalar

A = AngleLandmark


def _angles(t: float = 0.0, track_id: int = 0, **joints: float) -> Frame:
    return frame(track_id=track_id, t=t, features={Angles: scalar(Angles, {A[k]: v for k, v in joints.items()})})


def _head(node, n: int = 1) -> float:
    out = None
    for _ in range(n):
        out = node.update()
    return out[Angles][A.head]


class ChaseInterpolatorTest(unittest.TestCase):
    def test_update_before_set_is_none(self) -> None:
        node = AngleChaseInterpolator(ChaseInterpolatorSettings())
        self.assertIsNone(node.update())
        self.assertFalse(node.is_ready())

    def test_first_target_is_taken_immediately(self) -> None:
        node = AngleChaseInterpolator(ChaseInterpolatorSettings())
        node.set(_angles(head=0.7))
        self.assertAlmostEqual(_head(node), 0.7, places=6)

    def test_converges_on_a_fixed_target(self) -> None:
        node = AngleChaseInterpolator(ChaseInterpolatorSettings())
        node.set(_angles(head=0.0))
        node.update()
        node.set(_angles(head=1.0))
        first = _head(node)
        self.assertGreater(first, 0.0)
        self.assertLess(first, 1.0)
        self.assertAlmostEqual(_head(node, 300), 1.0, places=2)

    def test_angle_chase_takes_shortest_path_across_pi(self) -> None:
        node = AngleChaseInterpolator(ChaseInterpolatorSettings())
        node.set(_angles(head=math.pi - 0.1))
        node.update()
        node.set(_angles(head=-math.pi + 0.1))
        for i in range(120):
            head = _head(node)
            self.assertGreater(abs(head), math.pi - 0.3, f"update {i} swung through 0: {head}")
        self.assertAlmostEqual(head, -math.pi + 0.1, places=2)

    def test_values_stay_within_feature_range(self) -> None:
        settings = ChaseInterpolatorSettings()
        settings.responsiveness = 0.9
        settings.friction = 0.0
        node = SimilarityChaseInterpolator(settings)
        node.set(frame(features={Similarity: scalar(Similarity, {0: 0.0})}))
        node.update()
        node.set(frame(features={Similarity: scalar(Similarity, {0: 1.0})}))
        for _ in range(60):
            value = node.update()[Similarity][0]
            self.assertLessEqual(value, 1.0)
            self.assertGreaterEqual(value, 0.0)

    def test_missing_target_is_nan_with_zero_score(self) -> None:
        node = AngleChaseInterpolator(ChaseInterpolatorSettings())
        node.set(_angles(head=0.5, left_knee=0.2))
        node.update()
        node.set(_angles(left_knee=0.2))
        out = node.update()[Angles]
        self.assertTrue(math.isnan(out[A.head]))
        self.assertEqual(out.get_score(A.head), 0.0)
        self.assertTrue(out.validate()[0])

    def test_points_chase_per_coordinate(self) -> None:
        node = PointChaseInterpolator(ChaseInterpolatorSettings())
        node.set(frame(features={Points2D: points({PointLandmark.nose: (0.2, 0.4)})}))
        node.update()
        node.set(frame(features={Points2D: points({PointLandmark.nose: (0.6, 0.4)})}))
        for _ in range(300):
            out = node.update()[Points2D]
        np.testing.assert_allclose(out[PointLandmark.nose], (0.6, 0.4), atol=1e-2)

    def test_output_carries_last_frame_identity(self) -> None:
        node = AngleChaseInterpolator(ChaseInterpolatorSettings())
        node.set(frame(track_id=2, t=4.0, features={Angles: scalar(Angles, {A.head: 0.1})}))
        out = node.update()
        self.assertEqual((out.track_id, out.time_stamp), (2, 4.0))

    def test_reset_returns_to_not_ready(self) -> None:
        node = AngleChaseInterpolator(ChaseInterpolatorSettings())
        node.set(_angles(head=0.5))
        node.reset()
        self.assertIsNone(node.update())
        node.set(_angles(head=-0.5))
        self.assertAlmostEqual(_head(node), -0.5, places=6)


class LerpInterpolatorTest(unittest.TestCase):
    def test_reaches_target_in_one_input_interval(self) -> None:
        node = AngleLerpInterpolator(LerpInterpolatorSettings())         # 30 Hz in, 60 Hz out
        node.set(_angles(head=0.0))
        node.update()
        node.set(_angles(head=1.0))
        self.assertAlmostEqual(_head(node), 0.5, places=5)
        self.assertAlmostEqual(_head(node), 1.0, places=5)
        self.assertAlmostEqual(_head(node), 1.0, places=5)

    def test_angle_lerp_takes_shortest_path_across_pi(self) -> None:
        node = AngleLerpInterpolator(LerpInterpolatorSettings())
        node.set(_angles(head=math.pi - 0.1))
        node.update()
        node.set(_angles(head=-math.pi + 0.1))
        self.assertAlmostEqual(abs(_head(node)), math.pi, places=4)
        self.assertAlmostEqual(_head(node), -math.pi + 0.1, places=4)


class InterpolatorPipelineTest(unittest.TestCase):
    def test_merges_each_feature_from_its_own_node(self) -> None:
        pipeline = InterpolatorPipeline([
            AngleLerpInterpolator(LerpInterpolatorSettings()),
            AngleVelLerpInterpolator(LerpInterpolatorSettings()),
        ])
        pts = points({PointLandmark.nose: (0.5, 0.5)})

        def target(head: float, vel: float) -> Frame:
            return frame(track_id=1, t=2.0, features={
                Angles: scalar(Angles, {A.head: head}),
                AngleVelocity: scalar(AngleVelocity, {A.head: vel}),
                Points2D: pts,
            })

        pipeline.set(target(0.0, 0.0))
        pipeline.update()
        pipeline.set(target(1.0, 4.0))
        out = pipeline.update()
        self.assertAlmostEqual(out[Angles][A.head], 0.5, places=5)
        self.assertAlmostEqual(out[AngleVelocity][A.head], 2.0, places=5)
        self.assertIs(out[Points2D], pts)
        self.assertEqual(out.track_id, 1)

    def test_none_until_set_and_after_reset(self) -> None:
        pipeline = InterpolatorPipeline([AngleLerpInterpolator(LerpInterpolatorSettings())])
        self.assertIsNone(pipeline.update())
        pipeline.set(_angles(head=0.1))
        self.assertIsNotNone(pipeline.update())
        pipeline.reset()
        self.assertIsNone(pipeline.update())

    def test_empty_node_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InterpolatorPipeline([])


if __name__ == "__main__":
    unittest.main()
