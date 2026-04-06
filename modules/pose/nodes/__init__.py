"""Pose filter modules for processing and enriching pose data."""

from .extractors.AngleExtractor import          AngleExtractor, AngleExtractorSettings
from .extractors.AngleVelExtractor import       AngleVelExtractor, AngleVelExtractorSettings
from .extractors.AngleMotionExtractor import    AngleMotionExtractor, AngleMotionExtractorSettings
from .extractors.AngleSymExtractor import       AngleSymExtractor
from .extractors.AgeExtractor import            AgeExtractor
from .extractors.MotionTimeExtractor import     MotionTimeExtractor

from .applicators.SimilarityApplicator import   SimilarityApplicator, SimilarityApplicatorSettings
from .applicators.LeaderScoreApplicator import  LeaderScoreApplicator, LeaderScoreApplicatorSettings
from .applicators.MotionGateApplicator import   MotionGateApplicator, MotionGateApplicatorSettings

from .filters.ConfidenceFilters import          ConfidenceFilterSettings,     BBoxConfidenceFilter,   PointConfidenceFilter,  AngleConfidenceFilter,  AngleVelConfidenceFilter,   AngleSymConfFilter
from .filters.DualConfidenceFilters import      DualConfFilterSettings,       BBoxDualConfFilter,     PointDualConfFilter,    AngleDualConfFilter,    AngleVelDualConfFilter,     AngleSymDualConfidenceFilter
from .filters.Predictors import                 PredictorSettings,    BBoxPredictor,      PointPredictor,     AnglePredictor,     AngleVelPredictor,      AngleSymPredictor
from .filters.EmaSmoothers import               EmaSmootherSettings,  BBoxEmaSmoother,    PointEmaSmoother,   AngleEmaSmoother,   AngleVelEmaSmoother,    AngleSymEmaSmoother,    SimilarityEmaSmoother,  AngleMotionEmaSmoother
from .filters.EuroSmoothers import              EuroSmootherSettings, BBoxEuroSmoother,   PointEuroSmoother,  AngleEuroSmoother,  AngleVelEuroSmoother,   AngleSymEuroSmoother,   SimilarityEuroSmoother
from .filters.MovingAverageSmoothers import     MovingAverageSettings, WindowType, AngleMotionMovingAverageSmoother, SimilarityMovingAverageSmoother
from .filters.StickyFillers import              StickyFillerSettings, BBoxStickyFiller,   PointStickyFiller,  AngleStickyFiller,  AngleVelStickyFiller,   AngleSymStickyFiller,   SimilarityStickyFiller
from .filters.RateLimiters import               RateLimiterSettings,  BBoxRateLimiter,    PointRateLimiter,   AngleRateLimiter,   AngleVelRateLimiter,    AngleSymRateLimiter,    SimilarityRateLimiter,  AngleMotionRateLimiter
from .filters.EasingNode import                 EasingSettings,       EasingNode,         AngleMotionEasingNode
from .filters.TemporalFilters import            TemporalStabilizerSettings,   BBoxTemporalStabilizer, PointTemporalStabilizer,    AngleTemporalStabilizer,    AngleVelTemporalStabilizer, AngleSymTemporalStabilizer
from .filters.Validators import                 ValidatorSettings,    BBoxValidator,      PointValidator,     AngleValidator,     AngleVelValidator,      AngleSymValidator,      PoseValidator

from .interpolators.ChaseInterpolators import   ChaseInterpolatorSettings,    BBoxChaseInterpolator,  PointChaseInterpolator, AngleChaseInterpolator, AngleVelChaseInterpolator,  AngleSymChaseInterpolator,  SimilarityChaseInterpolator
from .interpolators.LerpInterpolators import    LerpInterpolatorSettings,     BBoxLerpInterpolator,   PointLerpInterpolator,  AngleLerpInterpolator,  AngleVelLerpInterpolator,   AngleSymLerpInterpolator

from modules.pose.nodes.windows.WindowNode import AngleMotionWindowNode, AngleSymmetryWindowNode, AngleVelocityWindowNode, AngleWindowNode, BBoxWindowNode, SimilarityWindowNode, WindowNode, WindowNodeSettings
from modules.pose.frame import FeatureWindow

# from .processors.ImageCropProcessor import      ImageCropProcessorConfig,   ImageCropProcessor
