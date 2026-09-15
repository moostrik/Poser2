"""End-to-end test of the pose stage chain, built from module nodes the way apps/white_space/main.py wires it.

CLEAN → SMOOTH → PREDICT → chase-interpolate → LERP extractors → motion gate, driven tick by tick with a moving
skeleton for two tracks: a confidence dropout on one wrist, and one track that leaves and comes back. App-local
nodes (playhead offset, ghoster) are left out; similarity is fed straight into the applicators instead of through
the threaded WindowSimilarity. Keep the node lists in step with main.py when its stages change.
"""

import logging
import math
import unittest

import numpy as np

from modules.pose.analytics import SimilarityResult
from modules.pose.features import (
    FEATURES, Age, AngleLandmark, AngleMotion, Angles, AngleSymmetry, AngleVelocity, ArmDeviation, Azimuth, BBox,
    BBoxAzimuth, LeaderScore, LegDeviation, MotionGate, MotionTime, PointLandmark, Points2D, Similarity, TorsoTilt,
)
from modules.pose.frame import FrameDict
from modules.pose.nodes import (
    AgeExtractor, AngleCalibrator, AngleCalibratorSettings, AngleChaseInterpolator, AngleEuroSmoother, AngleExtractor, AngleExtractorSettings,
    AngleMotionExtractor, AngleMotionExtractorSettings, AngleMotionMovingAverageSmoother, AnglePredictor,
    AngleStickyFiller, AngleSymExtractor, AngleVelChaseInterpolator, AngleVelEuroSmoother, AngleVelExtractor,
    AngleVelExtractorSettings, AngleVelPredictor, AngleVelStickyFiller, ArmDeviationExtractor,
    ArmDeviationExtractorSettings, AzimuthChaseInterpolator,
    AzimuthEuroSmoother, AzimuthExtractor, AzimuthPredictor, ChaseInterpolatorSettings,
    DualConfFilterSettings, EuroSmootherSettings, LeaderScoreApplicator,
    LegDeviationExtractor, LegDeviationExtractorSettings, MotionGateApplicator, MotionTimeExtractor,
    MovingAverageSettings, PointChaseInterpolator, PointDualConfFilter, PointEuroSmoother, PointPredictor,
    PointStickyFiller, PredictorSettings, SimilarityApplicator, SimilarityChaseInterpolator,
    SimilarityEuroSmoother, SimilarityStickyFiller, StickyFillerSettings, TorsoTiltExtractor,
    TorsoTiltExtractorSettings,
)
from modules.pose.trackers import FilterPipeline, FilterTracker, InterpolatorPipeline, InterpolatorTracker
from modules.utils import Rect

from ._builders import FPS, NUM_POSES, frame, points, skeleton_coords

TICKS = 90
DROPOUT = range(30, 35)        # track 0's left wrist falls below the confidence threshold
ABSENT = range(40, 50)         # track 1 is not detected
STAGES = ('raw', 'clean', 'smooth', 'predict', 'lerp', 'gate')


class _Settings:
    """One settings object per role, shared across tracks as the app's settings tree does."""

    def __init__(self) -> None:
        self.confidence = DualConfFilterSettings()
        self.point_sticky = StickyFillerSettings()
        self.angle_extractor = AngleExtractorSettings()
        self.angle_calibrator = AngleCalibratorSettings()
        self.velocity_extractor = AngleVelExtractorSettings()
        self.point_smoother = EuroSmootherSettings()
        self.velocity_smoother = EuroSmootherSettings()
        self.angle_smoother = EuroSmootherSettings()
        self.azimuth_smoother = EuroSmootherSettings()
        self.motion_extractor = AngleMotionExtractorSettings()
        self.motion_average = MovingAverageSettings()
        self.leg_deviation = LegDeviationExtractorSettings()
        self.arm_deviation = ArmDeviationExtractorSettings()
        self.torso_tilt = TorsoTiltExtractorSettings()
        self.similarity_smoother = EuroSmootherSettings()
        self.point_prediction = PredictorSettings()
        self.angle_prediction = PredictorSettings()
        self.velocity_prediction = PredictorSettings()
        self.azimuth_prediction = PredictorSettings()
        self.angle_sticky = StickyFillerSettings()
        self.similarity_sticky = StickyFillerSettings()
        self.velocity_sticky = StickyFillerSettings()
        self.point_interpolator = ChaseInterpolatorSettings()
        self.angle_interpolator = ChaseInterpolatorSettings()
        self.velocity_interpolator = ChaseInterpolatorSettings()
        self.similarity_interpolator = ChaseInterpolatorSettings()
        self.azimuth_interpolator = ChaseInterpolatorSettings()


class _Stages:
    def __init__(self) -> None:
        ps = self.settings = _Settings()
        tracks = range(NUM_POSES)

        self.clean = FilterTracker({i: FilterPipeline([
            PointDualConfFilter(ps.confidence),
            PointStickyFiller(ps.point_sticky),
            AzimuthExtractor(lambda _cam, x: x),
            AngleExtractor(ps.angle_extractor),
            AngleCalibrator(ps.angle_calibrator),
            AngleVelExtractor(ps.velocity_extractor),
        ]) for i in tracks})

        self.similarity_applicator = SimilarityApplicator()
        self.leader_applicator = LeaderScoreApplicator()
        self.smooth = FilterTracker({i: FilterPipeline([
            PointEuroSmoother(ps.point_smoother),
            AngleExtractor(ps.angle_extractor),
            AngleCalibrator(ps.angle_calibrator),
            AngleVelExtractor(ps.velocity_extractor),
            AngleVelEuroSmoother(ps.velocity_smoother),
            AngleEuroSmoother(ps.angle_smoother),
            AzimuthEuroSmoother(ps.azimuth_smoother),
            AngleMotionExtractor(ps.motion_extractor),
            AngleMotionMovingAverageSmoother(ps.motion_average),
            AngleSymExtractor(ps.leg_deviation),
            LegDeviationExtractor(ps.leg_deviation),
            ArmDeviationExtractor(ps.arm_deviation),
            TorsoTiltExtractor(ps.torso_tilt),
            MotionTimeExtractor(),
            AgeExtractor(),
            self.similarity_applicator,
            self.leader_applicator,
            SimilarityEuroSmoother(ps.similarity_smoother),
        ]) for i in tracks})

        self.predict = FilterTracker({i: FilterPipeline([
            PointPredictor(ps.point_prediction),
            AnglePredictor(ps.angle_prediction),
            AngleVelPredictor(ps.velocity_prediction),
            AzimuthPredictor(ps.azimuth_prediction),
            AngleStickyFiller(ps.angle_sticky),
            SimilarityStickyFiller(ps.similarity_sticky),
        ]) for i in tracks})

        self.interpolate = InterpolatorTracker({i: InterpolatorPipeline([
            PointChaseInterpolator(ps.point_interpolator),
            AngleChaseInterpolator(ps.angle_interpolator),
            AngleVelChaseInterpolator(ps.velocity_interpolator),
            SimilarityChaseInterpolator(ps.similarity_interpolator),
            AzimuthChaseInterpolator(ps.azimuth_interpolator),
        ]) for i in tracks})

        self.lerp = FilterTracker({i: FilterPipeline([
            AngleSymExtractor(ps.leg_deviation),
            LegDeviationExtractor(ps.leg_deviation),
            ArmDeviationExtractor(ps.arm_deviation),
            TorsoTiltExtractor(ps.torso_tilt),
            MotionTimeExtractor(),
            AgeExtractor(),
            AngleVelStickyFiller(ps.velocity_sticky),
            AngleVelEuroSmoother(ps.velocity_smoother),
            AngleMotionExtractor(ps.motion_extractor),
            AngleMotionMovingAverageSmoother(ps.motion_average),
        ]) for i in tracks})

        self.motion_gate = MotionGateApplicator()
        self.gate = FilterTracker({i: FilterPipeline([self.motion_gate]) for i in tracks})

    def tick(self, raw: FrameDict) -> dict[str, list[FrameDict]]:
        """One input tick: the 30 Hz stages once, then two 60 Hz interpolation updates."""
        out: dict[str, list[FrameDict]] = {'raw': [raw]}
        self._feed_similarity(raw)
        out['clean'] = [self.clean.process(raw)]
        out['smooth'] = [self.smooth.process(out['clean'][0])]
        out['predict'] = [self.predict.process(out['smooth'][0])]
        self.interpolate.set(out['predict'][0])
        out['lerp'], out['gate'] = [], []
        for _ in range(2):
            lerp = self.lerp.process(self.interpolate.update())
            self.motion_gate.set(lerp)
            out['lerp'].append(lerp)
            out['gate'].append(self.gate.process(lerp))
        return out

    def _feed_similarity(self, raw: FrameDict) -> None:
        """What WindowSimilarity publishes: nothing for fewer than two tracks, else a row per track."""
        similarity, leader = {}, {}
        for tid in (raw if len(raw) >= 2 else ()):
            values = np.full(NUM_POSES, np.nan, dtype=np.float32)
            scores = np.zeros(NUM_POSES, dtype=np.float32)
            lead = np.zeros(NUM_POSES, dtype=np.float32)
            for other in raw:
                if other != tid:
                    values[other], scores[other] = 0.6, 1.0
            similarity[tid] = Similarity(values, scores)
            leader[tid] = LeaderScore(lead, scores.copy())
        result = SimilarityResult(similarity, leader)
        self.similarity_applicator.set(result)
        self.leader_applicator.set(result)


def _raw(tick: int) -> FrameDict:
    t = tick / FPS
    frames: FrameDict = {}
    for tid in (0, 1):
        if tid == 1 and tick in ABSENT:
            continue
        swing = 0.8 * math.sin(2 * math.pi * 0.5 * t + tid)
        coords = skeleton_coords(left_elbow=swing, right_knee=0.3 * swing)
        scores = {PointLandmark.left_wrist: 0.1} if tid == 0 and tick in DROPOUT else None
        frames[tid] = frame(track_id=tid, cam_id=tid, t=t, features={
            Points2D: points(coords, scores),
            BBox: BBox.from_rect(Rect(0.2 + 0.3 * tid, 0.1, 0.25, 0.8)),
            BBoxAzimuth: BBoxAzimuth.from_value(0.5 + tid),
        })
    return frames


class PipelineStagesTest(unittest.TestCase):
    history: list[dict[str, list[FrameDict]]]
    errors: list[logging.LogRecord]

    @classmethod
    def setUpClass(cls) -> None:
        records: list[logging.LogRecord] = []

        class _Collect(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Collect(level=logging.ERROR)
        logger = logging.getLogger('modules.pose')
        logger.addHandler(handler)
        try:
            stages = _Stages()
            cls.history = [stages.tick(_raw(i)) for i in range(TICKS)]
        finally:
            logger.removeHandler(handler)
        cls.errors = records

    def _frames(self, stage: str):
        for tick, outputs in enumerate(self.history):
            for frames in outputs[stage]:
                for tid, f in frames.items():
                    yield tick, tid, f

    def test_no_errors_logged(self) -> None:
        self.assertEqual([r.getMessage() for r in self.errors], [])

    def test_every_stage_emits_contract_valid_features(self) -> None:
        for stage in STAGES:
            with self.subTest(stage=stage):
                for tick, tid, f in self._frames(stage):
                    for ft in FEATURES:
                        if ft in f:
                            ok, err = f[ft].validate()
                            self.assertTrue(ok, f"tick {tick} track {tid} {ft.__name__}: {err}")

    def test_identity_is_preserved(self) -> None:
        for stage in STAGES:
            with self.subTest(stage=stage):
                for tick, tid, f in self._frames(stage):
                    self.assertEqual(f.track_id, tid)
                    self.assertEqual(f.cam_id, tid)
                    self.assertAlmostEqual(f.time_stamp, tick / FPS, places=9)

    def test_every_present_track_comes_out_of_every_stage(self) -> None:
        for tick, outputs in enumerate(self.history):
            present = set(outputs['raw'][0])
            for stage in STAGES:
                for frames in outputs[stage]:
                    self.assertEqual(set(frames), present, f"tick {tick} stage {stage}")

    def test_smooth_stage_has_every_feature(self) -> None:
        expected = {Points2D, BBox, BBoxAzimuth, Azimuth, Angles, AngleVelocity, AngleMotion, AngleSymmetry, LegDeviation,
                    ArmDeviation, TorsoTilt, MotionTime, Age, Similarity, LeaderScore}
        _, _, f = list(self._frames('smooth'))[-1]
        self.assertEqual({ft for ft in FEATURES if ft in f}, expected)

    def test_azimuth_is_derived_at_clean_and_carried_to_lerp(self) -> None:
        # The skeleton's eyes sit on the box centre and the test projection is the identity, so the eye
        # azimuth equals the bbox azimuth wherever it exists.
        for _, _, f in self._frames('raw'):
            self.assertNotIn(Azimuth, f)
        for stage in ('clean', 'smooth', 'predict', 'lerp'):
            with self.subTest(stage=stage):
                for tick, tid, f in self._frames(stage):
                    self.assertAlmostEqual(f[Azimuth].value, 0.5 + tid, places=4, msg=f"tick {tick} track {tid}")

    def test_gate_stage_adds_motion_gate(self) -> None:
        _, _, f = list(self._frames('gate'))[-1]
        self.assertIn(MotionGate, f)

    def test_sticky_fill_bridges_wrist_dropout(self) -> None:
        for tick in DROPOUT:
            clean = self.history[tick]['clean'][0][0]
            wrist = clean[Points2D]
            self.assertTrue(bool(wrist.valid_mask[PointLandmark.left_wrist]), tick)
            self.assertEqual(wrist.get_score(PointLandmark.left_wrist), 0.0)
            self.assertFalse(math.isnan(clean[Angles][AngleLandmark.left_elbow]), tick)

    def test_elbow_swing_survives_to_lerp(self) -> None:
        elbows = [f[Angles][AngleLandmark.left_elbow] for _, tid, f in self._frames('lerp') if tid == 0]
        self.assertTrue(all(not math.isnan(v) for v in elbows))
        self.assertGreater(max(elbows) - min(elbows), 1.0)

    def test_returning_track_starts_its_age_over(self) -> None:
        before = self.history[ABSENT.start - 1]['lerp'][-1][1][Age].value
        after = self.history[ABSENT.stop + 2]['lerp'][-1][1][Age].value
        self.assertGreater(before, 1.0)
        self.assertLess(after, 0.2)

    def test_similarity_to_a_departed_track_decays_instead_of_holding(self) -> None:
        # With one track left, no similarity is published; the applicator falls back to zeros so the smoothed
        # and sticky-filled similarity decays toward 0 rather than holding the last 0.6.
        before = self.history[ABSENT.start - 1]['predict'][0][0][Similarity][1]
        during = self.history[ABSENT.stop - 1]['predict'][0][0][Similarity][1]
        self.assertAlmostEqual(before, 0.6, places=2)
        self.assertLess(during, 0.3)


if __name__ == "__main__":
    unittest.main()
