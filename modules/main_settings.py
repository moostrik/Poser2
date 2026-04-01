from modules.settings import Settings, NiceSettings, Field
from modules.oak import OakSettings
from modules.render import RenderSettings
from modules.inout import OscSoundConfig, ArtNetBarsSettings
from modules.tracker import OnePerCamTrackerConfig
from modules.pose import batch, nodes, trackers, Settings as PoseSettings
from modules.utils import TimerConfig


class InOutGroup(Settings):
    osc_sound:          OscSoundConfig
    artnet_1:           ArtNetBarsSettings
    artnet_2:           ArtNetBarsSettings
    artnet_3:           ArtNetBarsSettings

class SmoothedFeatureGroup(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    smoother            = nodes.EuroSmootherConfig(share=[frequency])
    interpolator        = nodes.ChaseInterpolatorConfig(share=[frequency.as_('input_frequency'), output_frequency])
    prediction          = nodes.PredictorConfig(share=[frequency])

class AngleFeatureGroup(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    smoother            = nodes.EuroSmootherConfig(share=[frequency])
    angle_vel_smoother  = nodes.EuroSmootherConfig(share=[frequency])
    interpolator        = nodes.ChaseInterpolatorConfig(share=[frequency.as_('input_frequency'), output_frequency])
    prediction          = nodes.PredictorConfig(share=[frequency])

class MotionGroup(Settings):
    extractor:          nodes.AngleMotionExtractorConfig
    moving_average:     nodes.MovingAverageConfig

class PoseGroup(Settings):
    max_poses:          Field[int]   = Field(3, min=1, max=16, access=Field.INIT)
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    pose                = PoseSettings(share=[max_poses])
    image_crop          = batch.ImageCropConfig(share=[max_poses])
    window_similarity   = batch.WindowSimilarityConfig(share=[max_poses])
    window_correlation  = batch.WindowCorrelationConfig(share=[max_poses])
    motion_gate         = nodes.MotionGateApplicatorConfig(share=[max_poses])
    confidence_filter:  nodes.DualConfFilterConfig
    bbox                = SmoothedFeatureGroup(share=[frequency, output_frequency])
    point               = SmoothedFeatureGroup(share=[frequency, output_frequency])
    angle               = AngleFeatureGroup(share=[frequency, output_frequency])
    similarity          = SmoothedFeatureGroup(share=[frequency, output_frequency])
    motion:             MotionGroup
    angle_sticky:       nodes.StickyFillerConfig
    similarity_sticky:  nodes.StickyFillerConfig
    angle_vel_sticky:   nodes.StickyFillerConfig
    window_raw:         trackers.WindowNodeConfig
    window_smooth:      trackers.WindowNodeConfig
    window_lerp:        trackers.WindowNodeConfig
    # rate_limiter:     nodes.RateLimiterConfig
    # easing:           nodes.EasingConfig

class TTGroup(Settings):
    timer:              TimerConfig
    tracker:            OnePerCamTrackerConfig

class MainSettings(Settings):
    num_players:        Field[int] = Field(3, access=Field.INIT, visible=False)
    camera:             OakSettings
    tt:                 TTGroup
    pose                = PoseGroup(share=[num_players.as_('max_poses')])
    render:             RenderSettings
    inout:              InOutGroup
    server:             NiceSettings
