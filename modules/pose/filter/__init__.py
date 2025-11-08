"""Pose filter modules for processing and enriching pose data."""

from .PoseCallbackFilter import                     PoseCallbackFilter
from .PoseBatchFilter import                        PoseBatchFilter
from .PoseBatchFilterPipeline import                PoseBatchFilterPipeline

from .extractor.PoseAngleExtractor import           PoseAngleExtractor
from .extractor.PoseDeltaExtractor import           PoseDeltaExtractor
from .extractor.PoseSymmetryExtractor import        PoseSymmetryExtractor
from .extractor.PoseMotionTimeAccumulator import    PoseMotionTimeAccumulator

from .general.PoseConfidenceFilter import           PoseConfidenceFilterConfig, PoseConfidenceFilter
from .general.PoseNanValidator import               PoseNanValidator
from .general.PoseRangeValidator import             PoseValidatorConfig, PoseRangeValidator

from .prediction.PoseChaseInterpolators import      PoseChaseInterpolatorConfig, PoseAngleChaseInterpolator, PoseDeltaChaseInterpolator, PosePointChaseInterpolator, PoseChaseInterpolator
from .prediction.PoseStickyFilters import           PoseStickyFilterConfig, PoseAngleStickyFilter, PoseDeltaStickyFilter, PosePointStickyFilter, PoseStickyFilter
from .prediction.PosePredictors import              PosePredictorConfig, PoseAnglePredictor, PoseDeltaPredictor, PosePointPredictor, PosePredictor
from .prediction.PoseSmoothers import               PoseSmootherConfig, PoseAngleSmoother, PoseDeltaSmoother, PosePointSmoother, PoseSmoother


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
    'PoseBatchFilterPipeline',

    'PoseAngleExtractor',
    'PoseDeltaExtractor',
    'PoseSymmetryExtractor',
    'PoseMotionTimeAccumulator',

    'PoseConfidenceFilter',
    'PoseConfidenceFilterConfig',

    'PoseValidatorConfig',
    'PoseNanValidator',
    'PoseRangeValidator',

    'PoseChaseInterpolatorConfig',
    'PoseAngleChaseInterpolator',
    'PoseDeltaChaseInterpolator',
    'PosePointChaseInterpolator',
    'PoseChaseInterpolator',

    'PoseStickyFilterConfig',
    'PoseAngleStickyFilter',
    'PoseDeltaStickyFilter',
    'PosePointStickyFilter',
    'PoseStickyFilter',

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

    # Old implementations (deprecated, aliased to avoid breaking changes)
    'OldPoseSmootherConfig',
    'OldPosePointSmoother',
    'OldPoseAngleSmoother',
    'OldPoseDeltaSmoother',
    'PoseBBoxSmoother',
    'PoseInterpolator',
    'PoseInterpolatorConfig',
]