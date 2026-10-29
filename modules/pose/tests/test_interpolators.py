"""Tests for the chase and lerp interpolator nodes and the InterpolatorPipeline that merges their outputs."""

import math
import unittest

import numpy as np

from modules.pose.features import AngleLandmark, Angles, AngleVelocity, Points2D, PointLandmark, Similarity
from modules.pose.frame import Frame
from modules.pose.nodes import (
    AngleChaseInterpolator, AngleLerpInterpolator, AngleVelocityLerpInterpolator, ChaseInterpolatorSettings,
    LerpInterpolatorSettings, PointChaseInterpolator, SimilarityChaseInterpolator, TickStep,
)
from modules.pose.trackers import InterpolatorPipeline

from ._builders import frame, points, scalar

A = AngleLandmark
TICK = 1.0 / 60.0          # the output clock of these tests: 60 Hz, two ticks per 30 Hz input frame


def _angles(t: float = 0.0, track_id: int = 0, **joints: float) -> Frame:
    return frame(track_id=track_id, t=t, features={Angles: scalar(Angles, {A[k]: v for k, v in joints.items()})})


class Ticker:
    """A node on a 60 Hz output clock: ``update()`` ticks it at the next stamp."""

    def __init__(self, node) -> None:
        self.node = node
        self.now = 0.0

    def set(self, pose: Frame) -> None:
        self.node.set(pose)

    def reset(self) -> None:
        self.node.reset()

    def update(self, dt: float = TICK) -> Frame | None:
        out = self.node.update(self.now)
        self.now += dt
        return out

    def head(self, n: int = 1) -> float:
        out = None
        for _ in range(n):
            out = self.update()
        return out[Angles][A.head]


def _chase(settings: ChaseInterpolatorSettings | None = None) -> Ticker:
    return Ticker(AngleChaseInterpolator(settings or ChaseInterpolatorSettings()))


class ChaseInterpolatorTest(unittest.TestCase):
    def test_update_before_set_is_none(self) -> None:
        node = _chase()
        self.assertIsNone(node.update())
        self.assertFalse(node.node.is_ready())

    def test_first_target_is_taken_immediately(self) -> None:
        node = _chase()
        node.set(_angles(head=0.7))
        self.assertAlmostEqual(node.head(), 0.7, places=6)

    def test_converges_on_a_fixed_target(self) -> None:
        node = _chase()
        node.set(_angles(head=0.0))
        node.update()
        node.set(_angles(head=1.0))
        first = node.head()
        self.assertGreater(first, 0.0)
        self.assertLess(first, 1.0)
        self.assertAlmostEqual(node.head(300), 1.0, places=2)

    def test_the_step_mode_says_how_far_a_late_tick_moves(self) -> None:
        def after(step: TickStep, gap: float) -> float:
            settings = ChaseInterpolatorSettings()
            settings.step = step
            node = _chase(settings)
            node.set(_angles(head=0.0))
            node.update()
            node.set(_angles(head=1.0))
            node.update(dt=gap)                                        # the next tick comes ``gap`` seconds later
            return node.head()

        self.assertAlmostEqual(after(TickStep.FIXED, 2.0), after(TickStep.FIXED, TICK), places=6)   # a tick is a tick
        self.assertGreater(after(TickStep.MEASURED, 2.0), after(TickStep.MEASURED, TICK))          # the whole gap
        self.assertGreater(after(TickStep.CAPPED, 1.0 / 30.0), after(TickStep.CAPPED, TICK))
        self.assertAlmostEqual(after(TickStep.CAPPED, 2.0), after(TickStep.CAPPED, 1.0 / 30.0), places=6)   # one frame at most

    def test_angle_chase_takes_shortest_path_across_pi(self) -> None:
        node = _chase()
        node.set(_angles(head=math.pi - 0.1))
        node.update()
        node.set(_angles(head=-math.pi + 0.1))
        for i in range(120):
            head = node.head()
            self.assertGreater(abs(head), math.pi - 0.3, f"update {i} swung through 0: {head}")
        self.assertAlmostEqual(head, -math.pi + 0.1, places=2)

    def test_values_stay_within_feature_range(self) -> None:
        settings = ChaseInterpolatorSettings()
        settings.responsiveness = 0.9
        settings.friction = 0.0
        node = Ticker(SimilarityChaseInterpolator(settings))
        node.set(frame(features={Similarity: scalar(Similarity, {0: 0.0})}))
        node.update()
        node.set(frame(features={Similarity: scalar(Similarity, {0: 1.0})}))
        for _ in range(60):
            value = node.update()[Similarity][0]
            self.assertLessEqual(value, 1.0)
            self.assertGreaterEqual(value, 0.0)

    def test_missing_target_is_nan_with_zero_score(self) -> None:
        node = _chase()
        node.set(_angles(head=0.5, left_knee=0.2))
        node.update()
        node.set(_angles(left_knee=0.2))
        out = node.update()[Angles]
        self.assertTrue(math.isnan(out[A.head]))
        self.assertEqual(out.get_score(A.head), 0.0)
        self.assertTrue(out.validate()[0])

    def test_points_chase_per_coordinate(self) -> None:
        node = Ticker(PointChaseInterpolator(ChaseInterpolatorSettings()))
        node.set(frame(features={Points2D: points({PointLandmark.nose: (0.2, 0.4)})}))
        node.update()
        node.set(frame(features={Points2D: points({PointLandmark.nose: (0.6, 0.4)})}))
        for _ in range(300):
            out = node.update()[Points2D]
        np.testing.assert_allclose(out[PointLandmark.nose], (0.6, 0.4), atol=1e-2)

    def test_output_carries_last_frame_identity(self) -> None:
        node = _chase()
        node.set(frame(track_id=2, t=4.0, features={Angles: scalar(Angles, {A.head: 0.1})}))
        out = node.update()
        self.assertEqual((out.track_id, out.timestamp), (2, 4.0))

    def test_reset_returns_to_not_ready(self) -> None:
        node = _chase()
        node.set(_angles(head=0.5))
        node.reset()
        self.assertIsNone(node.update())
        node.set(_angles(head=-0.5))
        self.assertAlmostEqual(node.head(), -0.5, places=6)


class LerpInterpolatorTest(unittest.TestCase):
    def test_reaches_target_in_one_input_interval(self) -> None:
        node = Ticker(AngleLerpInterpolator(LerpInterpolatorSettings()))      # 30 Hz in, 60 Hz ticks
        node.set(_angles(head=0.0))
        node.update()
        node.set(_angles(head=1.0))
        self.assertAlmostEqual(node.head(), 0.5, places=5)
        self.assertAlmostEqual(node.head(), 1.0, places=5)
        self.assertAlmostEqual(node.head(), 1.0, places=5)

    def test_angle_lerp_takes_shortest_path_across_pi(self) -> None:
        node = Ticker(AngleLerpInterpolator(LerpInterpolatorSettings()))
        node.set(_angles(head=math.pi - 0.1))
        node.update()
        node.set(_angles(head=-math.pi + 0.1))
        self.assertAlmostEqual(abs(node.head()), math.pi, places=4)
        self.assertAlmostEqual(node.head(), -math.pi + 0.1, places=4)


class InterpolatorPipelineTest(unittest.TestCase):
    def test_merges_each_feature_from_its_own_node(self) -> None:
        pipeline = InterpolatorPipeline([
            AngleLerpInterpolator(LerpInterpolatorSettings()),
            AngleVelocityLerpInterpolator(LerpInterpolatorSettings()),
        ])
        pts = points({PointLandmark.nose: (0.5, 0.5)})

        def target(head: float, vel: float) -> Frame:
            return frame(track_id=1, t=2.0, features={
                Angles: scalar(Angles, {A.head: head}),
                AngleVelocity: scalar(AngleVelocity, {A.head: vel}),
                Points2D: pts,
            })

        pipeline.set(target(0.0, 0.0))
        pipeline.update(0.0)
        pipeline.set(target(1.0, 4.0))
        out = pipeline.update(TICK)
        self.assertAlmostEqual(out[Angles][A.head], 0.5, places=5)
        self.assertAlmostEqual(out[AngleVelocity][A.head], 2.0, places=5)
        self.assertIs(out[Points2D], pts)
        self.assertEqual(out.track_id, 1)

    def test_the_frame_carries_the_time_of_the_tick_it_was_made_on(self) -> None:
        pipeline = InterpolatorPipeline([AngleLerpInterpolator(LerpInterpolatorSettings())])
        pipeline.set(_angles(t=2.0, head=0.1))
        self.assertEqual(pipeline.update(2.5).timestamp, 2.5)
        self.assertEqual(pipeline.update(2.52).timestamp, 2.52)        # not the target's 2.0, not repeated

    def test_none_until_set_and_after_reset(self) -> None:
        pipeline = InterpolatorPipeline([AngleLerpInterpolator(LerpInterpolatorSettings())])
        self.assertIsNone(pipeline.update(0.0))
        pipeline.set(_angles(head=0.1))
        self.assertIsNotNone(pipeline.update(0.0))
        pipeline.reset()
        self.assertIsNone(pipeline.update(0.0))

    def test_empty_node_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InterpolatorPipeline([])


if __name__ == "__main__":
    unittest.main()
