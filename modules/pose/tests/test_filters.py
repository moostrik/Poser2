"""Tests for the filter and applicator nodes wired into the pose stages: a contract sweep plus per-node behaviour."""

import math
import unittest

import numpy as np

from modules.pose.analytics import SimilarityResult
from modules.pose.features import (
    FEATURES, AngleLandmark, AngleMotion, Angles, AngleVelocity, Azimuth, BBox, LeaderScore, MotionGate,
    Points2D, PointLandmark, Similarity,
)
from modules.pose.frame import Frame
from modules.pose.nodes import (
    AgeExtractor, AngleEuroSmoother, AngleExtractor, AngleMotionExtractor, AngleMotionMovingAverageSmoother,
    AnglePredictor, AngleStickyFiller, AngleSymExtractor, AngleVelEuroSmoother, AngleVelExtractor,
    AngleVelPredictor, AngleVelStickyFiller, DualConfFilterSettings, EuroSmootherSettings, EyeAzimuthExtractor,
    FilterNode, LeaderScoreApplicator, LegDeviationExtractor, MotionGateApplicator, MotionTimeExtractor,
    MovingAverageSettings, PointDualConfFilter, PointEuroSmoother, PointPredictor, PointStickyFiller,
    PredictionMethod, PredictorSettings, SimilarityApplicator, SimilarityEuroSmoother, SimilarityStickyFiller,
    StickyFillerSettings, TorsoTiltExtractor, WindowType,
)
from modules.utils import Rect

from ._builders import FPS, NUM_POSES, frame, points, scalar, skeleton, wrap

A = AngleLandmark
P = PointLandmark


def _stage_nodes() -> list[FilterNode]:
    """Every FilterNode the app's stage pipelines use, with default settings."""
    return [
        PointDualConfFilter(DualConfFilterSettings()),
        PointStickyFiller(StickyFillerSettings()),
        AngleExtractor(),
        AngleVelExtractor(),
        PointEuroSmoother(EuroSmootherSettings()),
        AngleVelEuroSmoother(EuroSmootherSettings()),
        AngleEuroSmoother(EuroSmootherSettings()),
        AngleMotionExtractor(),
        AngleMotionMovingAverageSmoother(MovingAverageSettings()),
        AngleSymExtractor(),
        LegDeviationExtractor(),
        TorsoTiltExtractor(),
        MotionTimeExtractor(),
        AgeExtractor(),
        SimilarityApplicator(),
        LeaderScoreApplicator(),
        SimilarityEuroSmoother(EuroSmootherSettings()),
        PointPredictor(PredictorSettings()),
        AnglePredictor(PredictorSettings()),
        AngleVelPredictor(PredictorSettings()),
        AngleStickyFiller(StickyFillerSettings()),
        SimilarityStickyFiller(StickyFillerSettings()),
        AngleVelStickyFiller(StickyFillerSettings()),
        EyeAzimuthExtractor(lambda _cam, x: x),
        MotionGateApplicator(),
    ]


def _full_frame(i: int) -> Frame:
    """A realistic RAW-stage frame: moving skeleton, box and azimuth."""
    return frame(track_id=1, cam_id=0, t=i / FPS, features={
        Points2D: skeleton(left_elbow=0.05 * i),
        BBox: BBox.from_rect(Rect(0.3, 0.1, 0.4, 0.8)),
        Azimuth: Azimuth.from_value(0.5),
    })


def _assert_features_valid(test: unittest.TestCase, out: Frame) -> None:
    for ft in FEATURES:
        if ft in out:
            ok, err = out[ft].validate()
            test.assertTrue(ok, f"{ft.__name__}: {err}")


class FilterContractTest(unittest.TestCase):
    """Any node in a stage must accept empty and populated frames and emit contract-valid features."""

    def test_empty_frames(self) -> None:
        for node in _stage_nodes():
            with self.subTest(node=type(node).__name__):
                for i in range(3):
                    src = frame(track_id=2, cam_id=1, t=i / FPS)
                    out = node.process(src)
                    self.assertEqual((out.track_id, out.cam_id, out.time_stamp), (2, 1, i / FPS))
                    _assert_features_valid(self, out)

    def test_populated_frames_keep_identity_and_input(self) -> None:
        for node in _stage_nodes():
            with self.subTest(node=type(node).__name__):
                for i in range(5):
                    src = _full_frame(i)
                    pts, n = src[Points2D], len(src)
                    out = node.process(src)
                    self.assertEqual((out.track_id, out.cam_id, out.time_stamp), (1, 0, i / FPS))
                    self.assertIs(src[Points2D], pts)
                    self.assertEqual(len(src), n)
                    _assert_features_valid(self, out)

    def test_reset_is_safe_at_any_time(self) -> None:
        for node in _stage_nodes():
            with self.subTest(node=type(node).__name__):
                node.reset()
                node.process(_full_frame(0))
                node.reset()
                _assert_features_valid(self, node.process(_full_frame(1)))


def _conf_frame(wrist_score: float) -> Frame:
    """Points with the nose always confident and the left wrist at the given score."""
    return frame(features={Points2D: points({P.nose: (0.5, 0.2), P.left_wrist: (0.3, 0.6)},
                                            scores={P.left_wrist: wrist_score})})


class DualConfFilterTest(unittest.TestCase):
    def _filter(self, rescale: bool = False) -> PointDualConfFilter:
        settings = DualConfFilterSettings()
        settings.threshold_low = 0.3
        settings.threshold_high = 0.5
        settings.rescale_scores = rescale
        return PointDualConfFilter(settings)

    def _wrist_visible(self, f: PointDualConfFilter, score: float) -> bool:
        return bool(f.process(_conf_frame(score))[Points2D].valid_mask[P.left_wrist])

    def test_hysteresis(self) -> None:
        f = self._filter()
        sequence = [(0.4, False), (0.6, True), (0.4, True), (0.3, True), (0.2, False), (0.4, False), (0.5, True)]
        for score, visible in sequence:
            with self.subTest(score=score, expected=visible):
                self.assertEqual(self._wrist_visible(f, score), visible)

    def test_hidden_elements_are_nan_with_zero_score(self) -> None:
        out = self._filter().process(_conf_frame(0.1))[Points2D]
        self.assertTrue(np.all(np.isnan(out[P.left_wrist])))
        self.assertEqual(out.get_score(P.left_wrist), 0.0)

    def test_scores_kept_without_rescale(self) -> None:
        out = self._filter(rescale=False).process(_conf_frame(0.6))[Points2D]
        self.assertAlmostEqual(out.get_score(P.left_wrist), 0.6, places=6)

    def test_scores_rescaled_from_low_threshold(self) -> None:
        f = self._filter(rescale=True)
        self.assertAlmostEqual(f.process(_conf_frame(0.6))[Points2D].get_score(P.left_wrist), (0.6 - 0.3) / 0.7, places=5)
        self.assertAlmostEqual(f.process(_conf_frame(1.0))[Points2D].get_score(P.nose), 1.0, places=5)

    def test_frame_without_valid_points_resets_state(self) -> None:
        f = self._filter()
        self.assertTrue(self._wrist_visible(f, 0.6))
        f.process(frame())                                  # no points at all
        self.assertFalse(self._wrist_visible(f, 0.4))

    def test_reset_clears_state(self) -> None:
        f = self._filter()
        self._wrist_visible(f, 0.6)
        f.reset()
        self.assertFalse(self._wrist_visible(f, 0.4))


def _angle_frame(t: float = 0.0, score: float = 1.0, **joints: float) -> Frame:
    return frame(t=t, features={Angles: scalar(Angles, {A[k]: v for k, v in joints.items()}, score=score)})


class StickyFillerTest(unittest.TestCase):
    def _filler(self, **fields: bool) -> AngleStickyFiller:
        settings = StickyFillerSettings()
        for name, value in fields.items():
            setattr(settings, name, value)
        return AngleStickyFiller(settings)

    def test_holds_last_valid_value_with_zero_score(self) -> None:
        f = self._filler()
        f.process(_angle_frame(score=0.8, head=0.5))
        out = f.process(_angle_frame(left_knee=0.1))[Angles]
        self.assertAlmostEqual(out[A.head], 0.5, places=6)
        self.assertEqual(out.get_score(A.head), 0.0)
        self.assertAlmostEqual(out[A.left_knee], 0.1, places=6)

    def test_hold_scores_keeps_last_score(self) -> None:
        f = self._filler(hold_scores=True)
        f.process(_angle_frame(score=0.8, head=0.5))
        out = f.process(_angle_frame())[Angles]
        self.assertAlmostEqual(out.get_score(A.head), 0.8, places=6)

    def test_new_valid_value_replaces_held_one(self) -> None:
        f = self._filler()
        f.process(_angle_frame(head=0.5))
        f.process(_angle_frame(head=0.9))
        self.assertAlmostEqual(f.process(_angle_frame())[Angles][A.head], 0.9, places=6)

    def test_never_seen_stays_nan(self) -> None:
        out = self._filler().process(_angle_frame(head=0.5))[Angles]
        self.assertTrue(math.isnan(out[A.left_knee]))

    def test_init_to_zero(self) -> None:
        out = self._filler(init_to_zero=True).process(_angle_frame())[Angles]
        self.assertTrue(np.all(out.values == 0.0))
        self.assertTrue(np.all(out.scores == 0.0))

    def test_disabled_passes_frame_through(self) -> None:
        src = _angle_frame()
        self.assertIs(self._filler(enabled=False).process(src), src)

    def test_reset_forgets_held_value(self) -> None:
        f = self._filler()
        f.process(_angle_frame(head=0.5))
        f.reset()
        self.assertTrue(math.isnan(f.process(_angle_frame())[Angles][A.head]))


class EuroSmootherTest(unittest.TestCase):
    def test_constant_input_is_a_fixed_point(self) -> None:
        f = AngleEuroSmoother(EuroSmootherSettings())
        for i in range(10):
            out = f.process(_angle_frame(i / FPS, head=1.0))
        self.assertAlmostEqual(out[Angles][A.head], 1.0, places=5)

    def test_smooths_a_step(self) -> None:
        f = AngleEuroSmoother(EuroSmootherSettings())
        for i in range(10):
            f.process(_angle_frame(i / FPS, head=0.0))
        stepped = f.process(_angle_frame(10 / FPS, head=1.0))[Angles][A.head]
        self.assertGreater(stepped, 0.0)
        self.assertLess(stepped, 1.0)

    def test_angle_step_across_pi_goes_the_short_way(self) -> None:
        f = AngleEuroSmoother(EuroSmootherSettings())
        for i in range(10):
            f.process(_angle_frame(i / FPS, head=math.pi - 0.1))
        for i in range(10, 20):
            head = f.process(_angle_frame(i / FPS, head=-math.pi + 0.1))[Angles][A.head]
            self.assertGreater(abs(head), math.pi - 0.15, f"frame {i} swung through 0: {head}")

    def test_nan_passes_through_with_zero_score(self) -> None:
        f = AngleEuroSmoother(EuroSmootherSettings())
        f.process(_angle_frame(0.0, head=0.3))
        out = f.process(_angle_frame(1 / FPS, left_knee=0.3))[Angles]
        self.assertTrue(math.isnan(out[A.head]))
        self.assertEqual(out.get_score(A.head), 0.0)

    def test_reappearing_value_starts_fresh(self) -> None:
        f = AngleEuroSmoother(EuroSmootherSettings())
        for i in range(5):
            f.process(_angle_frame(i / FPS, head=0.0))
        f.process(_angle_frame(5 / FPS))
        self.assertAlmostEqual(f.process(_angle_frame(6 / FPS, head=1.0))[Angles][A.head], 1.0, places=5)

    def test_reset_forgets_history(self) -> None:
        f = AngleEuroSmoother(EuroSmootherSettings())
        for i in range(10):
            f.process(_angle_frame(i / FPS, head=0.0))
        f.reset()
        self.assertAlmostEqual(f.process(_angle_frame(10 / FPS, head=1.0))[Angles][A.head], 1.0, places=5)

    def test_points_are_clamped_to_range(self) -> None:
        f = PointEuroSmoother(EuroSmootherSettings())
        out = f.process(frame(features={Points2D: points({P.nose: (2.5, -1.5)})}))[Points2D]
        np.testing.assert_allclose(out[P.nose], (2.0, -1.0))


class PredictorTest(unittest.TestCase):
    def _predictor(self, method: PredictionMethod) -> AnglePredictor:
        settings = PredictorSettings()
        settings.method = method
        return AnglePredictor(settings)

    def test_first_sample_predicts_itself(self) -> None:
        out = self._predictor(PredictionMethod.LINEAR).process(_angle_frame(head=0.4))
        self.assertAlmostEqual(out[Angles][A.head], 0.4, places=6)

    def test_linear_extrapolates_one_step(self) -> None:
        p = self._predictor(PredictionMethod.LINEAR)
        p.process(_angle_frame(head=0.0))
        p.process(_angle_frame(head=0.1))
        self.assertAlmostEqual(p.process(_angle_frame(head=0.2))[Angles][A.head], 0.3, places=5)

    def test_quadratic_is_exact_on_a_ramp_once_settled(self) -> None:
        p = self._predictor(PredictionMethod.QUADRATIC)
        for v in (0.0, 0.1, 0.2):
            p.process(_angle_frame(head=v))
        self.assertAlmostEqual(p.process(_angle_frame(head=0.3))[Angles][A.head], 0.4, places=5)

    def test_prediction_wraps_across_pi(self) -> None:
        p = self._predictor(PredictionMethod.LINEAR)
        p.process(_angle_frame(head=math.pi - 0.2))
        out = p.process(_angle_frame(head=math.pi - 0.05))[Angles][A.head]
        self.assertAlmostEqual(out, wrap(math.pi + 0.1), places=5)

    def test_missing_value_predicts_nan_with_zero_score(self) -> None:
        p = self._predictor(PredictionMethod.QUADRATIC)
        p.process(_angle_frame(head=0.1, left_knee=0.1))
        out = p.process(_angle_frame(left_knee=0.2))[Angles]
        self.assertTrue(math.isnan(out[A.head]))
        self.assertEqual(out.get_score(A.head), 0.0)

    def test_reset_clears_history(self) -> None:
        p = self._predictor(PredictionMethod.LINEAR)
        p.process(_angle_frame(head=0.0))
        p.reset()
        self.assertAlmostEqual(p.process(_angle_frame(head=0.5))[Angles][A.head], 0.5, places=6)


class MovingAverageTest(unittest.TestCase):
    def _smoother(self, size: int, kind: WindowType) -> AngleMotionMovingAverageSmoother:
        settings = MovingAverageSettings()
        settings.window_size = size
        settings.window_type = kind
        return AngleMotionMovingAverageSmoother(settings)

    def _feed(self, s: AngleMotionMovingAverageSmoother, value: float) -> float:
        return s.process(frame(features={AngleMotion: AngleMotion.from_value(value)}))[AngleMotion].value

    def test_uniform_window_mean(self) -> None:
        s = self._smoother(3, WindowType.UNIFORM)
        for v in (0.3, 0.6):
            self._feed(s, v)
        self.assertAlmostEqual(self._feed(s, 0.9), 0.6, places=5)
        self.assertAlmostEqual(self._feed(s, 0.0), 0.5, places=5)     # 0.3 left the window

    def test_triangular_weights_newest_most(self) -> None:
        s = self._smoother(3, WindowType.TRIANGULAR)
        for v in (0.0, 0.0):
            self._feed(s, v)
        self.assertAlmostEqual(self._feed(s, 0.6), 0.6 * 3 / 6, places=5)

    def test_partial_window_averages_what_it_has(self) -> None:
        s = self._smoother(30, WindowType.UNIFORM)
        self._feed(s, 0.2)
        self.assertAlmostEqual(self._feed(s, 0.4), 0.3, places=5)

    def test_missing_samples_are_skipped(self) -> None:
        s = self._smoother(3, WindowType.UNIFORM)
        self._feed(s, 0.2)
        s.process(frame())
        self.assertAlmostEqual(self._feed(s, 0.4), 0.3, places=5)

    def test_reset_clears_window(self) -> None:
        s = self._smoother(3, WindowType.UNIFORM)
        self._feed(s, 0.9)
        s.reset()
        self.assertAlmostEqual(self._feed(s, 0.1), 0.1, places=5)


def _similarity(values: dict[int, float]) -> Similarity:
    return scalar(Similarity, values)


class ApplicatorTest(unittest.TestCase):
    def test_similarity_applicator_stamps_own_row(self) -> None:
        app = SimilarityApplicator()
        mine, theirs = _similarity({1: 0.7}), _similarity({0: 0.7})
        app.set(SimilarityResult(similarity={0: mine, 1: theirs}, leader_score={}))
        self.assertIs(app.process(frame(track_id=0))[Similarity], mine)
        self.assertIs(app.process(frame(track_id=1))[Similarity], theirs)

    def test_similarity_applicator_unknown_track_gets_valid_zeros(self) -> None:
        app = SimilarityApplicator()
        app.set(SimilarityResult(similarity={0: _similarity({1: 0.7})}, leader_score={}))
        out = app.process(frame(track_id=3))[Similarity]
        self.assertTrue(np.all(out.values == 0.0))
        self.assertTrue(np.all(out.scores == 1.0))

    def test_similarity_applicator_replaces_stale_result(self) -> None:
        app = SimilarityApplicator()
        app.set(SimilarityResult(similarity={0: _similarity({1: 0.7})}, leader_score={}))
        app.set(SimilarityResult(similarity={}, leader_score={}))
        self.assertTrue(np.all(app.process(frame(track_id=0))[Similarity].values == 0.0))

    def test_leader_applicator_stamps_own_row_or_leaves_frame(self) -> None:
        app = LeaderScoreApplicator()
        leader = LeaderScore(np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32), np.array([0, 1, 0, 0], dtype=np.float32))
        app.set(SimilarityResult(similarity={}, leader_score={0: leader}))
        self.assertIs(app.process(frame(track_id=0))[LeaderScore], leader)
        src = frame(track_id=2)
        self.assertIs(app.process(src), src)

    def test_motion_gate_is_self_motion_and_pairwise_products(self) -> None:
        app = MotionGateApplicator()
        motions = {0: 0.5, 2: 0.4, 3: math.nan}
        app.set({tid: frame(track_id=tid, features={AngleMotion: AngleMotion.from_value(m)}) for tid, m in motions.items()})
        gate = app.process(frame(track_id=0))[MotionGate]
        np.testing.assert_allclose(gate.values, [0.5, 0.0, 0.2, 0.0], atol=1e-6)
        np.testing.assert_allclose(gate.scores, [1.0, 0.0, 1.0, 1.0])

    def test_motion_gate_ignores_track_ids_beyond_max_poses(self) -> None:
        app = MotionGateApplicator()
        app.set({0: frame(track_id=0, features={AngleMotion: AngleMotion.from_value(0.5)}),
                 NUM_POSES: frame(track_id=NUM_POSES, features={AngleMotion: AngleMotion.from_value(0.5)})})
        gate = app.process(frame(track_id=0))[MotionGate]
        self.assertEqual(len(gate), NUM_POSES)
        self.assertEqual(app.process(frame(track_id=NUM_POSES))[MotionGate].values[0], 0.25)

    def test_motion_gate_absent_track_leaves_frame(self) -> None:
        app = MotionGateApplicator()
        app.set({})
        src = frame(track_id=1)
        self.assertIs(app.process(src), src)


class AngleVelocityChainTest(unittest.TestCase):
    def test_velocity_sticky_holds_and_zeroes_score(self) -> None:
        f = AngleVelStickyFiller(StickyFillerSettings())
        f.process(frame(features={AngleVelocity: scalar(AngleVelocity, {A.head: 2.0})}))
        out = f.process(frame())[AngleVelocity]
        self.assertAlmostEqual(out[A.head], 2.0, places=6)
        self.assertEqual(out.get_score(A.head), 0.0)


if __name__ == "__main__":
    unittest.main()
