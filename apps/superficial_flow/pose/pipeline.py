"""Superficial Flow's pose pipeline: the stages, their steps and their clock domains.

The frame domain (the camera's frame bang) smooths and predicts the tracklet boxes, runs the crops through
the pose model and the RAW, CLEAN and SMOOTH stages on the model's thread; the render tick drives LERP, where the
predict interpolators show each feature where its fit puts it at the tick's time.
"""

from __future__ import annotations

from modules.pose import nodes
from modules.pose.pipeline import AsyncInput, AsyncOutput, AsyncStep, Domain, FrameSource, GroupStep, PipelineSpec, PoseStep, StageSpec

from ..settings import PoseGroup, Stage


def pipeline_spec(ps: PoseGroup, frame_source: FrameSource, crops: AsyncInput, poses: AsyncOutput) -> PipelineSpec:
    """The pipeline on the app's pose settings ``ps``; ``frame_source`` is the tracker's frame source and
    ``crops`` the crop step and ``poses`` the pose model
    it feeds."""
    return PipelineSpec([
        StageSpec(Stage.RAW, frame_source, [
            PoseStep(nodes.BBoxEuroSmoother, ps.bbox.smoother),
            PoseStep(nodes.BBoxPredictor, ps.bbox.prediction),
            AsyncStep(crops, poses),
        ], window=ps.window_raw),

        StageSpec(Stage.CLEAN, Stage.RAW, [
            PoseStep(nodes.PointDualConfidenceFilter, ps.point.confidence_filter),
            PoseStep(nodes.AngleExtractor, ps.angle_extractor),
            PoseStep(nodes.AngleCalibrator),
            PoseStep(nodes.AngleVelocityExtractor, ps.velocity.extractor),
        ], window=ps.window_clean),

        StageSpec(Stage.SMOOTH, Stage.CLEAN, [
            PoseStep(nodes.PointEuroSmoother, ps.point.smoother),
            PoseStep(nodes.AngleExtractor, ps.angle_extractor),
            PoseStep(nodes.AngleCalibrator),
            PoseStep(nodes.AngleVelocityExtractor, ps.velocity.extractor),
            PoseStep(nodes.AngleVelocityEuroSmoother, ps.velocity.smoother),
            PoseStep(nodes.AngleEuroSmoother, ps.angle.smoother),
            PoseStep(nodes.AngleMotionExtractor, ps.motion.extractor),
            PoseStep(nodes.AngleMotionMovingAverageSmoother, ps.motion.moving_average),
            PoseStep(nodes.AngleSymmetryExtractor),
            PoseStep(nodes.MotionTimeExtractor),
            PoseStep(nodes.AgeExtractor),
            PoseStep(nodes.AngleStickyFiller, ps.angle.sticky),
        ], window=ps.window_smooth),

        StageSpec(Stage.LERP, Stage.SMOOTH, [
            PoseStep(nodes.AngleSymmetryExtractor),
            PoseStep(nodes.MotionTimeExtractor),
            PoseStep(nodes.AgeExtractor),
            PoseStep(nodes.AngleVelocityStickyFiller, ps.velocity.sticky),
            PoseStep(nodes.AngleVelocityEuroSmoother, ps.velocity.smoother),
            PoseStep(nodes.AngleMotionExtractor, ps.motion.extractor),
            PoseStep(nodes.AngleMotionMovingAverageSmoother, ps.motion.moving_average),
            GroupStep(nodes.MotionGateExtractor()),
        ], domain=Domain.OUTPUT, interpolators=[
            PoseStep(nodes.BBoxPredictInterpolator, ps.bbox.interpolator),
            PoseStep(nodes.PointPredictInterpolator, ps.point.interpolator),
            PoseStep(nodes.AnglePredictInterpolator, ps.angle.interpolator),
            PoseStep(nodes.AngleVelocityPredictInterpolator, ps.velocity.interpolator),
        ], window=ps.window_lerp),
    ])
