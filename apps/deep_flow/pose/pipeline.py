"""Deep Flow's pose pipeline: the stages, their steps and their clock domains.

The frame domain (the camera's frame bang) smooths and predicts the tracklet boxes, runs the crops through
the pose model and the RAW, CLEAN, SMOOTH and PREDICT stages on the model's thread; the render tick drives
LERP.
"""

from __future__ import annotations

from modules.pose import nodes
from modules.pose.pipeline import AsyncNode, AsyncStep, Domain, FrameSource, GroupStep, PipelineSpec, PoseStep, StageSpec

from ..settings import PoseGroup, Stage


def pipeline_spec(ps: PoseGroup, tracklet_poses: FrameSource, pose_inference: AsyncNode) -> PipelineSpec:
    """The pipeline on the app's pose settings ``ps``; ``tracklet_poses`` is the camera source and
    ``pose_inference`` the crop and pose model."""
    return PipelineSpec([
        StageSpec(Stage.RAW, tracklet_poses, [
            PoseStep(nodes.BBoxEuroSmoother, ps.bbox.smoother),
            PoseStep(nodes.BBoxPredictor, ps.bbox.prediction),
            AsyncStep(pose_inference),
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
        ], window=ps.window_smooth),

        StageSpec(Stage.PREDICT, Stage.SMOOTH, [
            PoseStep(nodes.PointPredictor, ps.point.prediction),
            PoseStep(nodes.AnglePredictor, ps.angle.prediction),
            PoseStep(nodes.AngleVelocityPredictor, ps.velocity.prediction),
            PoseStep(nodes.AngleStickyFiller, ps.angle.sticky),
        ], window=ps.window_predict),

        StageSpec(Stage.LERP, Stage.PREDICT, [
            PoseStep(nodes.AngleSymmetryExtractor),
            PoseStep(nodes.MotionTimeExtractor),
            PoseStep(nodes.AgeExtractor),
            PoseStep(nodes.AngleVelocityStickyFiller, ps.velocity.sticky),
            PoseStep(nodes.AngleVelocityEuroSmoother, ps.velocity.smoother),
            PoseStep(nodes.AngleMotionExtractor, ps.motion.extractor),
            PoseStep(nodes.AngleMotionMovingAverageSmoother, ps.motion.moving_average),
            GroupStep(nodes.MotionGateExtractor()),
        ], domain=Domain.OUTPUT, interpolators=[
            PoseStep(nodes.BBoxChaseInterpolator, ps.bbox.interpolator),
            PoseStep(nodes.PointChaseInterpolator, ps.point.interpolator),
            PoseStep(nodes.AngleChaseInterpolator, ps.angle.interpolator),
            PoseStep(nodes.AngleVelocityChaseInterpolator, ps.velocity.interpolator),
        ], window=ps.window_lerp),
    ])
