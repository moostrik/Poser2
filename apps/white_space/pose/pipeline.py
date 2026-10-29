"""White Space's pose pipeline: the stages, their steps and their clock domains.

The frame domain (the camera's frame bang) runs the tracklet poses through the pose model and the RAW,
CLEAN, SMOOTH and PREDICT stages on the model's thread. The light tick drives LERP: the interpolated
players and the dummy meet there, and everything a person has is stamped on both. ``docs/POSE_INSTRUMENT.md``
says what the stages mean to the show and ``docs/SIMILARITY.md`` what the similarity steps compute.
"""

from __future__ import annotations

from typing import Callable

from modules.pose import features, nodes
from modules.pose.pipeline import AsyncNode, AsyncStep, Domain, FrameSource, GroupStep, Merge, PipelineSpec, PoseStep, \
                                  StageSpec

from ..settings import PoseGroup, Stage
from .dummy import Dummy
from .neutral_weight import NeutralWeight
from .playhead_offset import PlayheadOffset, PlayheadOffsetExtractor


def pipeline_spec(ps: PoseGroup, tracklet_poses: FrameSource, pose_inference: AsyncNode, dummy: Dummy,
                  column_to_azimuth: Callable[[int, float], float], playhead_phase: Callable[[], float]) -> PipelineSpec:
    """The pipeline on the app's pose settings ``ps``; ``tracklet_poses`` is the camera source, ``pose_inference``
    the crop and pose model, ``column_to_azimuth`` the tracker's geometry for the eye azimuth and ``playhead_phase``
    the board's playhead for the LERP stage's playhead offset."""
    return PipelineSpec([
        StageSpec(Stage.RAW, tracklet_poses, [
            AsyncStep(pose_inference),
        ], window=ps.window_raw),

        StageSpec(Stage.CLEAN, Stage.RAW, [
            PoseStep(nodes.PointDualConfidenceFilter, ps.point.confidence),
            PoseStep(nodes.PointStickyFiller, ps.point.sticky),
            # Here, not later: the keypoints are still normalised in the crop they came from, so the box's
            # jitter cancels out of the eye azimuth. A held eye moves with the box.
            PoseStep(nodes.AzimuthExtractor, column_to_azimuth),
            PoseStep(nodes.AngleExtractor, ps.angle_extractor),
            PoseStep(nodes.AngleCalibrator, ps.angle_calibrator),
            PoseStep(nodes.AngleVelocityExtractor, ps.velocity.extractor),
        ], window=ps.window_clean),

        StageSpec(Stage.SMOOTH, Stage.CLEAN, [
            PoseStep(nodes.PointEuroSmoother, ps.point.smoother),
            PoseStep(nodes.AngleExtractor, ps.angle_extractor),
            PoseStep(nodes.AngleCalibrator, ps.angle_calibrator),
            PoseStep(nodes.AngleVelocityExtractor, ps.velocity.extractor),
            PoseStep(nodes.AngleVelocityEuroSmoother, ps.velocity.smoother),
            PoseStep(nodes.AngleEuroSmoother, ps.angle.smoother),
            PoseStep(nodes.AzimuthEuroSmoother, ps.azimuth.smoother),
            PoseStep(nodes.DistanceEuroSmoother, ps.distance.smoother),
            PoseStep(nodes.AngleMotionExtractor, ps.motion.extractor),
            PoseStep(nodes.AngleMotionMovingAverageSmoother, ps.motion.moving_average),
            PoseStep(nodes.AngleSymmetryExtractor, ps.leg_deviation_extractor),
            PoseStep(nodes.LegDeviationExtractor, ps.leg_deviation_extractor),
            PoseStep(nodes.ArmDeviationExtractor, ps.arm_deviation_extractor),
            PoseStep(nodes.ArmTravelExtractor, ps.arm_travel_extractor),
            PoseStep(nodes.ShoulderDifferenceExtractor),
            PoseStep(nodes.TorsoTiltExtractor, ps.torso_tilt_extractor),
            PoseStep(nodes.MotionTimeExtractor),
            PoseStep(nodes.AgeExtractor),
            # The group steps: the posture similarity (the pairwise distance of the poses as a 0..1 similarity)
            # stamped on them; where presence is known, the sticky filler bridges a present pair's gap (a departed
            # player's slot stays NaN); the neutral weight scales a pair by how far both are out of neutral (their
            # arm deviation, this frame's); the hold stamps the rows one frame late, the lag the show was tuned
            # with. Then the rows are smoothed.
            GroupStep(nodes.PostureSimilarityExtractor(ps.similarity.posture)),
            GroupStep(nodes.SimilarityStickyFiller(ps.similarity.sticky)),
            GroupStep(NeutralWeight(ps.similarity.neutral_weight)),
            GroupStep(nodes.OneFrameHold(features.Similarity)),
            PoseStep(nodes.SimilarityEuroSmoother, ps.similarity.smoother),
        ], window=ps.window_smooth),

        StageSpec(Stage.PREDICT, Stage.SMOOTH, [
            PoseStep(nodes.PointPredictor, ps.point.prediction),
            PoseStep(nodes.AnglePredictor, ps.angle.prediction),
            PoseStep(nodes.AngleVelocityPredictor, ps.velocity.prediction),
            PoseStep(nodes.AzimuthPredictor, ps.azimuth.prediction),
            PoseStep(nodes.AngleStickyFiller, ps.angle.sticky),
        ], window=ps.window_predict),

        # The light tick: the players interpolated from PREDICT, the dummy polled, then the LERP filters stamp on
        # both everything a person has (the playhead offset, the symmetries, the leg deviation, the bend).
        StageSpec(Stage.LERP, Merge(Stage.PREDICT, dummy), [
            # First, before the extractors: every angle-derived measure (arm travel, leg deviation) inherits
            # the cap, and the sound's angle messages read the same values.
            PoseStep(nodes.AngleRateLimiter, ps.angle.rate_limiter),
            PoseStep(nodes.AngleSymmetryExtractor, ps.leg_deviation_extractor),
            PoseStep(nodes.LegDeviationExtractor, ps.leg_deviation_extractor),
            PoseStep(nodes.ArmDeviationExtractor, ps.arm_deviation_extractor),
            PoseStep(nodes.ArmTravelExtractor, ps.arm_travel_extractor),
            PoseStep(nodes.ShoulderDifferenceExtractor),
            PoseStep(nodes.TorsoTiltExtractor, ps.torso_tilt_extractor),
            PoseStep(nodes.MotionTimeExtractor),
            PoseStep(nodes.AgeExtractor),
            PoseStep(nodes.AngleVelocityStickyFiller, ps.velocity.sticky),
            PoseStep(nodes.AngleVelocityEuroSmoother, ps.velocity.smoother),
            PoseStep(nodes.AngleMotionExtractor, ps.motion.extractor),
            PoseStep(nodes.AngleMotionMovingAverageSmoother, ps.motion.moving_average),
            PoseStep(PlayheadOffsetExtractor, playhead_phase),
            GroupStep(nodes.MotionGateExtractor()),
        ], domain=Domain.OUTPUT, interpolators=[
            PoseStep(nodes.PointChaseInterpolator, ps.point.interpolator),
            PoseStep(nodes.AngleChaseInterpolator, ps.angle.interpolator),
            PoseStep(nodes.AngleVelocityChaseInterpolator, ps.velocity.interpolator),
            PoseStep(nodes.SimilarityChaseInterpolator, ps.similarity.interpolator),
            PoseStep(nodes.AzimuthChaseInterpolator, ps.azimuth.interpolator),
            PoseStep(nodes.DistanceChaseInterpolator, ps.distance.interpolator),
        # The LERP window also holds the app-local playhead offset (the only stage where it is stamped) so the
        # data layers can graph it.
        ], window=ps.window_lerp, window_features=features.SCALAR_FEATURES + [PlayheadOffset]),
    ])
