"""Pose filter modules for processing and enriching pose data."""

from .nodes import                              NodeBase, FilterNode, GroupNode, InterpolatorNode

from .extractors.angle_extractor import         AngleExtractor, AngleExtractorSettings
from .extractors.angle_velocity_extractor import AngleVelocityExtractor, AngleVelocityExtractorSettings
from .extractors.angle_motion_extractor import  AngleMotionExtractor, AngleMotionExtractorSettings
from .extractors.angle_symmetry_extractor import AngleSymmetryExtractor
from .extractors.age_extractor import           AgeExtractor
from .extractors.motion_time_extractor import   MotionTimeExtractor
from .extractors.leg_deviation_extractor import LegDeviationExtractor, LegDeviationExtractorSettings
from .extractors.arm_deviation_extractor import ArmDeviationExtractor, ArmDeviationExtractorSettings
from .extractors.arm_travel_extractor import    ArmTravelExtractor, ArmTravelExtractorSettings
from .extractors.shoulder_difference_extractor import ShoulderDifferenceExtractor
from .extractors.torso_tilt_extractor import    TorsoTiltExtractor, TorsoTiltExtractorSettings
from .extractors.azimuth_extractor import       AzimuthExtractor

from .group.posture_similarity_extractor import PostureSimilarityExtractor
from .group.similarity_sticky_filler import     SimilarityStickyFiller, SimilarityStickyFillerSettings
from .group.motion_gate_extractor import        MotionGateExtractor
from .group.similarity_result_hold import       SimilarityResultHold
from .group.group_sync_gate import              GroupSyncGate, GroupSyncSettings

from .filters.angle_calibrator import           AngleCalibrator, AngleCalibratorSettings
from .filters.confidence_filters import         ConfidenceFilterSettings,     BBoxConfidenceFilter,   PointConfidenceFilter,  AngleConfidenceFilter,  AngleVelocityConfidenceFilter,   AngleSymmetryConfidenceFilter
from .filters.dual_confidence_filters import    DualConfidenceFilterSettings, BBoxDualConfidenceFilter, PointDualConfidenceFilter, AngleDualConfidenceFilter, AngleVelocityDualConfidenceFilter, AngleSymmetryDualConfidenceFilter
from .filters.predictors import                 PredictorSettings,    PredictionMethod,   BBoxPredictor,     PointPredictor,     AnglePredictor,     AngleVelocityPredictor,      AngleSymmetryPredictor,      AzimuthPredictor
from .filters.ema_smoothers import              EmaSmootherSettings,  BBoxEmaSmoother,    PointEmaSmoother,   AngleEmaSmoother,   AngleVelocityEmaSmoother,    AngleSymmetryEmaSmoother,    SimilarityEmaSmoother,  AngleMotionEmaSmoother
from .filters.euro_smoothers import             EuroSmootherSettings, BBoxEuroSmoother,   PointEuroSmoother,  AngleEuroSmoother,  AngleVelocityEuroSmoother,   AngleSymmetryEuroSmoother,   SimilarityEuroSmoother, AzimuthEuroSmoother,    DistanceEuroSmoother
from .filters.moving_average_smoothers import   MovingAverageSmootherSettings, WindowType, AngleMotionMovingAverageSmoother, SimilarityMovingAverageSmoother
from .filters.sticky_fillers import             StickyFillerSettings, BBoxStickyFiller,   PointStickyFiller,  AngleStickyFiller,  AngleVelocityStickyFiller,   AngleSymmetryStickyFiller
from .filters.rate_limiters import              RateLimiterSettings,  BBoxRateLimiter,    PointRateLimiter,   AngleRateLimiter,   AngleVelocityRateLimiter,    AngleSymmetryRateLimiter,    SimilarityRateLimiter,  AngleMotionRateLimiter
from .filters.easing_node import                EasingNodeSettings,   EasingNode,         AngleMotionEasingNode
from .filters.temporal_stabilizers import       TemporalStabilizerSettings,   BBoxTemporalStabilizer, PointTemporalStabilizer,    AngleTemporalStabilizer,    AngleVelocityTemporalStabilizer, AngleSymmetryTemporalStabilizer

from .interpolators.chase_interpolators import  ChaseInterpolatorSettings,    BBoxChaseInterpolator,  PointChaseInterpolator, AngleChaseInterpolator, AngleVelocityChaseInterpolator,  AngleSymmetryChaseInterpolator,  SimilarityChaseInterpolator,    AzimuthChaseInterpolator,   DistanceChaseInterpolator
from .interpolators.lerp_interpolators import   LerpInterpolatorSettings,     BBoxLerpInterpolator,   PointLerpInterpolator,  AngleLerpInterpolator,  AngleVelocityLerpInterpolator,   AngleSymmetryLerpInterpolator

from ..frame import                             FeatureWindow
