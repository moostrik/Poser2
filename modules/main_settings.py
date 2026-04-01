from modules.settings import Settings, NiceSettings, Field
from modules.oak import OakSettings
from modules.render import RenderSettings
from modules.inout import OscSoundConfig, ArtNetBarsSettings
from modules.tracker import OnePerCamTrackerConfig
from modules.pose import batch, nodes, Settings as PoseSettings
from modules.utils import TimerConfig


class InOutGroup(Settings):
    osc_sound:          OscSoundConfig
    artnet_1:           ArtNetBarsSettings
    artnet_2:           ArtNetBarsSettings
    artnet_3:           ArtNetBarsSettings

class SmoothedFeatureGroup(Settings):
    smoother:           nodes.EuroSmootherConfig
    interpolator:       nodes.ChaseInterpolatorConfig
    prediction:         nodes.PredictorConfig

class AngleFeatureGroup(Settings):
    smoother:           nodes.EuroSmootherConfig
    angle_vel_smoother: nodes.EuroSmootherConfig
    interpolator:       nodes.ChaseInterpolatorConfig
    prediction:         nodes.PredictorConfig

class MotionGroup(Settings):
    extractor:          nodes.AngleMotionExtractorConfig
    moving_average:     nodes.MovingAverageConfig

class PoseGroup(Settings):
    pose:               PoseSettings
    window_similarity:  batch.WindowSimilarityConfig
    window_correlation: batch.WindowCorrelationConfig
    bbox:               SmoothedFeatureGroup
    point:              SmoothedFeatureGroup
    angle:              AngleFeatureGroup
    similarity:         SmoothedFeatureGroup
    motion:             MotionGroup

class TTGroup(Settings):
    timer:              TimerConfig
    tracker:            OnePerCamTrackerConfig

class MainSettings(Settings):
    num_players:        Field[int] = Field(3, access=Field.INIT, visible=False)
    camera:             OakSettings
    tt:                 TTGroup
    pose:               PoseGroup
    render:             RenderSettings
    inout:              InOutGroup
    server:             NiceSettings
