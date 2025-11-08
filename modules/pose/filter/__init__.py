"""Pose filter modules for processing and enriching pose data."""

from .PoseCallbackFilter import                     PoseCallbackFilter
from .PoseBatchFilter import                        PoseFilterTracker
from .PoseBatchFilterPipeline import                PoseFilterPipelineTracker

from .extractor.PoseAngleExtractor import           PoseAngleExtractor
from .extractor.PoseDeltaExtractor import           PoseDeltaExtractor
from .extractor.PoseSymmetryExtractor import        PoseSymmetryExtractor
from .extractor.PoseMotionTimeAccumulator import    PoseMotionTimeAccumulator

from .general.PoseValidators import                 PoseValidatorConfig, PoseNanValidator, PoseRangeValidator, PoseScoreValidator, PoseValidator

from .general.PoseChaseInterpolators import         PoseChaseInterpolatorConfig, PoseAngleChaseInterpolator, PoseDeltaChaseInterpolator, PosePointChaseInterpolator, PoseChaseInterpolator
from .general.PoseConfidenceFilters import          PoseConfidenceFilterConfig, PoseAngleConfidenceFilter, PoseDeltaConfidenceFilter, PosePointConfidenceFilter, PoseConfidenceFilter
from .general.PosePredictors import                 PosePredictorConfig, PoseAnglePredictor, PoseDeltaPredictor, PosePointPredictor, PosePredictor
from .general.PoseSmoothers import                  PoseSmootherConfig, PoseAngleSmoother, PoseDeltaSmoother, PosePointSmoother, PoseSmoother
from .general.PoseStickyFillers import              PoseStickyFillerConfig, PoseAngleStickyFiller, PoseDeltaStickyFiller, PosePointStickyFiller, PoseStickyFiller


from .smooth.PoseSmootherBase import                PoseSmootherConfig as OldPoseSmootherConfig
from .smooth.PosePointSmoother import               PosePointSmoother as OldPosePointSmoother
from .smooth.PoseAngleSmoother import               PoseAngleSmoother as OldPoseAngleSmoother
from .smooth.PoseDeltaSmoother import               PoseDeltaSmoother as OldPoseDeltaSmoother
from .smooth.PoseBBoxSmoother import                PoseBBoxSmoother
from .interpolation.PoseInterpolator import         PoseInterpolator
from .interpolation.PoseInterpolatorConfig import   PoseInterpolatorConfig

__all__: list[str] = [
    'PoseCallbackFilter',
    'PoseBatchFilter',
    'PoseFilterPipelineTracker',

    'PoseAngleExtractor',
    'PoseDeltaExtractor',
    'PoseSymmetryExtractor',
    'PoseMotionTimeAccumulator',

    'PoseValidatorConfig',
    'PoseNanValidator',
    'PoseRangeValidator',
    'PoseScoreValidator',
    'PoseValidator',

    'PoseChaseInterpolatorConfig',
    'PoseAngleChaseInterpolator',
    'PoseDeltaChaseInterpolator',
    'PosePointChaseInterpolator',
    'PoseChaseInterpolator',

    'PoseConfidenceFilterConfig',
    'PoseAngleConfidenceFilter',
    'PosePointConfidenceFilter',
    'PoseDeltaConfidenceFilter',
    'PoseConfidenceFilter',

    'PosePredictorConfig',
    'PoseAnglePredictor',
    'PoseDeltaPredictor',
    'PosePointPredictor',
    'PosePredictor',

    'PoseSmootherConfig',
    'PoseAngleSmoother',
    'PosePointSmoother',
    'PoseDeltaSmoother',
    'PoseSmoother',

    'PoseStickyFillerConfig',
    'PoseAngleStickyFiller',
    'PoseDeltaStickyFiller',
    'PosePointStickyFiller',
    'PoseStickyFiller',

    # Old implementations (deprecated, aliased to avoid breaking changes)
    'OldPoseSmootherConfig',
    'OldPosePointSmoother',
    'OldPoseAngleSmoother',
    'OldPoseDeltaSmoother',
    'PoseBBoxSmoother',
    'PoseInterpolator',
    'PoseInterpolatorConfig',
]