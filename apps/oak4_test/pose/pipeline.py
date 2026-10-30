"""The OAK4 test app's pose pipeline: the stages, their steps and their clock domains.

The camera runs the pose model, so the frame domain starts from poses that already carry their keypoints
(`PosesFromTracklets` fills them in the host crop's convention): no crop and no pose model here. The camera's
frame bang drives RAW, CLEAN and SMOOTH; the render tick drives LERP, where the predict interpolators show each
feature where its fit puts it at the tick's time.
"""

from __future__ import annotations

from modules.pose import nodes
from modules.pose.pipeline import Domain, FrameSource, PipelineSpec, PoseStep, StageSpec

from ..settings import PoseGroup, Stage


def pipeline_spec(ps: PoseGroup, tracklet_poses: FrameSource) -> PipelineSpec:
    """The pipeline on the app's pose settings ``ps``; ``tracklet_poses`` is the camera source."""
    return PipelineSpec([
        StageSpec(Stage.RAW, tracklet_poses, [], window=ps.window_raw),

        StageSpec(Stage.CLEAN, Stage.RAW, [
            PoseStep(nodes.PointDualConfidenceFilter, ps.point.confidence_filter),
            PoseStep(nodes.AngleExtractor, ps.angle_extractor),
        ], window=ps.window_clean),

        StageSpec(Stage.SMOOTH, Stage.CLEAN, [
            PoseStep(nodes.BBoxEuroSmoother, ps.bbox.smoother),
            PoseStep(nodes.PointEuroSmoother, ps.point.smoother),
            PoseStep(nodes.AngleExtractor, ps.angle_extractor),
            PoseStep(nodes.AngleEuroSmoother, ps.angle.smoother),
            PoseStep(nodes.AgeExtractor),
        ], window=ps.window_smooth),

        StageSpec(Stage.LERP, Stage.SMOOTH, [
            PoseStep(nodes.AgeExtractor),
        ], domain=Domain.OUTPUT, interpolators=[
            PoseStep(nodes.BBoxPredictInterpolator, ps.bbox.interpolator),
            PoseStep(nodes.PointPredictInterpolator, ps.point.interpolator),
            PoseStep(nodes.AnglePredictInterpolator, ps.angle.interpolator),
        ], window=ps.window_lerp),
    ])
