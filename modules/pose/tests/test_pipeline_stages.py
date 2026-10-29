"""End-to-end test of the pose stage chain, built by the pipeline builder from module nodes the way the apps'
pipeline files declare it (apps/hd_trio/pose/pipeline.py with White Space's azimuth and posture steps).

RAW → CLEAN → SMOOTH → PREDICT in the frame domain, then the crossing into LERP on the tick, driven tick by
tick with a moving skeleton for two tracks: a confidence dropout on one wrist, and one track that leaves and
comes back. App-local nodes (the playhead offset) are left out; similarity is fed into the result hold
instead of through the threaded WindowSimilarity. Keep the node lists in step with the apps' pipeline files
when their stages change.
"""

import logging
import math
import unittest
from enum import IntEnum
from functools import partial

import numpy as np

from modules.pose.analytics import SimilarityResult
from modules.pose.features import (
    FEATURES, Age, AngleLandmark, AngleMotion, Angles, AngleSymmetry, AngleVelocity, ArmDeviation, ArmTravel,
    Azimuth, BBox, BBoxAzimuth, LeaderScore, LegDeviation, MotionGate, MotionTime, PointLandmark, Points2D,
    ShoulderDifference, Similarity, TorsoTilt,
)
from modules.pose.frame import FrameDict
from modules.pose.nodes import (
    AgeExtractor, AngleCalibrator, AngleCalibratorSettings, AngleChaseInterpolator, AngleEuroSmoother, AngleExtractor, AngleExtractorSettings,
    AngleMotionExtractor, AngleMotionExtractorSettings, AngleMotionMovingAverageSmoother, AnglePredictor,
    AngleStickyFiller, AngleSymmetryExtractor, AngleVelocityChaseInterpolator, AngleVelocityEuroSmoother, AngleVelocityExtractor,
    AngleVelocityExtractorSettings, AngleVelocityPredictor, AngleVelocityStickyFiller, ArmDeviationExtractor,
    ArmDeviationExtractorSettings, ArmTravelExtractor, ArmTravelExtractorSettings, AzimuthChaseInterpolator,
    AzimuthEuroSmoother, AzimuthExtractor, AzimuthPredictor, ChaseInterpolatorSettings,
    DualConfidenceFilterSettings, EuroSmootherSettings,
    LegDeviationExtractor, LegDeviationExtractorSettings, MotionGateExtractor, MotionTimeExtractor,
    MovingAverageSmootherSettings, PointChaseInterpolator, PointDualConfidenceFilter, PointEuroSmoother, PointPredictor,
    PointStickyFiller, PredictorSettings, ShoulderDifferenceExtractor, SimilarityChaseInterpolator,
    SimilarityEuroSmoother, SimilarityResultHold, SimilarityStickyFiller, SimilarityStickyFillerSettings,
    StickyFillerSettings, TorsoTiltExtractor,
    TorsoTiltExtractorSettings,
)
from modules.pose.pipeline import Domain, GroupStep, PipelineSpec, PosePipeline, PoseStep, StageSpec
from modules.pose.window import WindowNodeSettings
from modules.utils import Rect

from ._builders import FPS, NUM_POSES, frame, points, skeleton_coords

TICKS = 90
DROPOUT = range(30, 35)        # track 0's left wrist falls below the confidence threshold
ABSENT = range(40, 50)         # track 1 is not detected
STAGES = ('raw', 'clean', 'smooth', 'predict', 'lerp')


class Stage(IntEnum):
    RAW = 0
    CLEAN = 1
    SMOOTH = 2
    PREDICT = 3
    LERP = 4


class _Source:
    """The camera source: pushes a tick's raw frames into the frame domain."""

    def __init__(self, track_ids: list[int]) -> None:
        self.track_ids = track_ids
        self._callbacks: list = []

    def add_frames_callback(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, frames: FrameDict) -> None:
        for callback in self._callbacks:
            callback(frames)


class _Settings:
    """One settings object per role, shared across tracks as the app's settings tree does."""

    def __init__(self) -> None:
        self.confidence = DualConfidenceFilterSettings()
        self.point_sticky = StickyFillerSettings()
        self.angle_extractor = AngleExtractorSettings()
        self.angle_calibrator = AngleCalibratorSettings()
        self.velocity_extractor = AngleVelocityExtractorSettings()
        self.point_smoother = EuroSmootherSettings()
        self.velocity_smoother = EuroSmootherSettings()
        self.angle_smoother = EuroSmootherSettings()
        self.azimuth_smoother = EuroSmootherSettings()
        self.motion_extractor = AngleMotionExtractorSettings()
        self.motion_average = MovingAverageSmootherSettings()
        self.leg_deviation = LegDeviationExtractorSettings()
        self.arm_deviation = ArmDeviationExtractorSettings()
        self.arm_travel = ArmTravelExtractorSettings()
        self.torso_tilt = TorsoTiltExtractorSettings()
        self.similarity_smoother = EuroSmootherSettings()
        self.point_prediction = PredictorSettings()
        self.angle_prediction = PredictorSettings()
        self.velocity_prediction = PredictorSettings()
        self.azimuth_prediction = PredictorSettings()
        self.angle_sticky = StickyFillerSettings()
        self.similarity_sticky = SimilarityStickyFillerSettings()
        self.velocity_sticky = StickyFillerSettings()
        self.point_interpolator = ChaseInterpolatorSettings()
        self.angle_interpolator = ChaseInterpolatorSettings()
        self.velocity_interpolator = ChaseInterpolatorSettings()
        self.similarity_interpolator = ChaseInterpolatorSettings()
        self.azimuth_interpolator = ChaseInterpolatorSettings()


class _Stages:
    def __init__(self) -> None:
        ps = self.settings = _Settings()
        self.source = _Source(list(range(NUM_POSES)))
        self.similarity_hold = SimilarityResultHold()
        self.pipeline = PosePipeline(PipelineSpec([
            StageSpec(Stage.RAW, self.source, []),

            StageSpec(Stage.CLEAN, Stage.RAW, [
                PoseStep(PointDualConfidenceFilter, ps.confidence),
                PoseStep(PointStickyFiller, ps.point_sticky),
                PoseStep(AzimuthExtractor, lambda _cam, x: x),
                PoseStep(AngleExtractor, ps.angle_extractor),
                PoseStep(AngleCalibrator, ps.angle_calibrator),
                PoseStep(AngleVelocityExtractor, ps.velocity_extractor),
            ]),

            StageSpec(Stage.SMOOTH, Stage.CLEAN, [
                PoseStep(PointEuroSmoother, ps.point_smoother),
                PoseStep(AngleExtractor, ps.angle_extractor),
                PoseStep(AngleCalibrator, ps.angle_calibrator),
                PoseStep(AngleVelocityExtractor, ps.velocity_extractor),
                PoseStep(AngleVelocityEuroSmoother, ps.velocity_smoother),
                PoseStep(AngleEuroSmoother, ps.angle_smoother),
                PoseStep(AzimuthEuroSmoother, ps.azimuth_smoother),
                PoseStep(AngleMotionExtractor, ps.motion_extractor),
                PoseStep(AngleMotionMovingAverageSmoother, ps.motion_average),
                PoseStep(AngleSymmetryExtractor, ps.leg_deviation),
                PoseStep(LegDeviationExtractor, ps.leg_deviation),
                PoseStep(ArmDeviationExtractor, ps.arm_deviation),
                PoseStep(ArmTravelExtractor, ps.arm_travel),
                PoseStep(ShoulderDifferenceExtractor),
                PoseStep(TorsoTiltExtractor, ps.torso_tilt),
                PoseStep(MotionTimeExtractor),
                PoseStep(AgeExtractor),
                # The group steps: the held analytics result stamped, a present pair's gap bridged; then smoothed.
                GroupStep(self.similarity_hold),
                GroupStep(SimilarityStickyFiller(ps.similarity_sticky)),
                PoseStep(SimilarityEuroSmoother, ps.similarity_smoother),
            ], window=WindowNodeSettings()),

            StageSpec(Stage.PREDICT, Stage.SMOOTH, [
                PoseStep(PointPredictor, ps.point_prediction),
                PoseStep(AnglePredictor, ps.angle_prediction),
                PoseStep(AngleVelocityPredictor, ps.velocity_prediction),
                PoseStep(AzimuthPredictor, ps.azimuth_prediction),
                PoseStep(AngleStickyFiller, ps.angle_sticky),
            ]),

            StageSpec(Stage.LERP, Stage.PREDICT, [
                PoseStep(AngleSymmetryExtractor, ps.leg_deviation),
                PoseStep(LegDeviationExtractor, ps.leg_deviation),
                PoseStep(ArmDeviationExtractor, ps.arm_deviation),
                PoseStep(ArmTravelExtractor, ps.arm_travel),
                PoseStep(ShoulderDifferenceExtractor),
                PoseStep(TorsoTiltExtractor, ps.torso_tilt),
                PoseStep(MotionTimeExtractor),
                PoseStep(AgeExtractor),
                PoseStep(AngleVelocityStickyFiller, ps.velocity_sticky),
                PoseStep(AngleVelocityEuroSmoother, ps.velocity_smoother),
                PoseStep(AngleMotionExtractor, ps.motion_extractor),
                PoseStep(AngleMotionMovingAverageSmoother, ps.motion_average),
                GroupStep(MotionGateExtractor()),
            ], domain=Domain.OUTPUT, interpolators=[
                PoseStep(PointChaseInterpolator, ps.point_interpolator),
                PoseStep(AngleChaseInterpolator, ps.angle_interpolator),
                PoseStep(AngleVelocityChaseInterpolator, ps.velocity_interpolator),
                PoseStep(SimilarityChaseInterpolator, ps.similarity_interpolator),
                PoseStep(AzimuthChaseInterpolator, ps.azimuth_interpolator),
            ]),
        ]))
        self.out: dict[str, list[FrameDict]] = {}
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self._record, stage.name.lower()))

    def _record(self, name: str, frames: FrameDict) -> None:
        self.out[name].append(frames)

    def tick(self, raw: FrameDict, now: float) -> dict[str, list[FrameDict]]:
        """One input tick: the 30 Hz stages once on the bang, then two 60 Hz ticks of the output domain."""
        self.out = {name: [] for name in STAGES}
        self._feed_similarity(raw)
        self.source.emit(raw)
        for _ in range(2):
            self.pipeline.update(Domain.OUTPUT, now)
        return self.out

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
        self.similarity_hold.set(SimilarityResult(similarity, leader))


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
            cls.history = [stages.tick(_raw(i), i / FPS) for i in range(TICKS)]
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

    def test_every_stage_emits_once_per_bang_and_lerp_once_per_tick(self) -> None:
        for outputs in self.history:
            for stage in STAGES:
                self.assertEqual(len(outputs[stage]), 2 if stage == 'lerp' else 1, stage)

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
                    self.assertAlmostEqual(f.timestamp, tick / FPS, places=9)

    def test_every_present_track_comes_out_of_every_stage(self) -> None:
        for tick, outputs in enumerate(self.history):
            present = set(outputs['raw'][0])
            for stage in STAGES:
                for frames in outputs[stage]:
                    self.assertEqual(set(frames), present, f"tick {tick} stage {stage}")

    def test_smooth_stage_has_every_feature(self) -> None:
        expected = {Points2D, BBox, BBoxAzimuth, Azimuth, Angles, AngleVelocity, AngleMotion, AngleSymmetry, LegDeviation,
                    ArmDeviation, ArmTravel, ShoulderDifference, TorsoTilt, MotionTime, Age, Similarity, LeaderScore}
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

    def test_lerp_stage_adds_the_motion_gate(self) -> None:
        _, _, f = list(self._frames('lerp'))[-1]
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

    def test_similarity_to_a_departed_track_is_no_data(self) -> None:
        # With one track left, no similarity is published and nothing holds the last 0.6: the slot of the
        # departed track is NaN through PREDICT and LERP, not a value toward someone who is gone.
        before = self.history[ABSENT.start - 1]['predict'][0][0][Similarity][1]
        during = self.history[ABSENT.stop - 1]['predict'][0][0][Similarity][1]
        at_lerp = self.history[ABSENT.stop - 1]['lerp'][-1][0][Similarity][1]
        self.assertAlmostEqual(before, 0.6, places=2)
        self.assertTrue(math.isnan(during))
        self.assertTrue(math.isnan(at_lerp))


if __name__ == "__main__":
    unittest.main()
