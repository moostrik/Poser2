# TODO
# better group lists
# log visible in gui
# why are groups after other settings
# what to do with mainsettings in tabs
# refactor settings
# share fps / frequencies across all settings
# update all settings to use new system


from modules.settings import Settings, NiceSettings, Field
from modules.oak import CameraSettings, FrameType, CoderFormat, SimulatorSettings, RecorderSettings, SyncSettings
from modules.render import RenderSettings
from modules.inout import OscSoundSettings, ArtNetBarsSettings
from modules.tracker import OnePerCamTrackerSettings
from modules.pose import batch, nodes, trackers
from modules.pose.batch.model_types import ModelType
from modules.utils import TimerSettings


class OakGroup(Settings):
    num_cameras:        Field[int]               = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    fps:                Field[float]             = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    yolo:               Field[bool]              = Field(True, access=Field.INIT, description="Enable YOLO person detection")
    color:              Field[bool]              = Field(True, access=Field.INIT, description="Enable color capture")
    square:             Field[bool]              = Field(True, access=Field.INIT, description="Use square aspect ratio")
    stereo:             Field[bool]              = Field(False, access=Field.INIT, description="Enable stereo mode")
    hd_ready:           Field[bool]              = Field(False, access=Field.INIT, description="Use HD resolution")
    sim_enabled:        Field[bool]              = Field(False, access=Field.INIT, description="Enable simulation mode")
    model_path:         Field[str]               = Field("models", access=Field.INIT, visible=False, description="Model files directory")
    video_path:         Field[str]               = Field("recordings", access=Field.INIT, visible=False, description="Video recordings directory")
    temp_path:          Field[str]               = Field("temp", access=Field.INIT, visible=False, description="Temporary files directory")
    video_format:       Field[CoderFormat]       = Field(CoderFormat.H264, access=Field.INIT, description="Video format")
    video_frame_types:  Field[list[FrameType]]   = Field([FrameType.VIDEO], access=Field.INIT, description="Frame types to record")

    cameras: list[CameraSettings]       = CameraSettings(count=num_cameras, share=[fps, color, square, stereo, yolo, hd_ready, sim_enabled, model_path])  # type: ignore[assignment]
    simulator     = SimulatorSettings(share=[video_path, video_format, video_frame_types, num_cameras, color, square, stereo])
    recorder      = RecorderSettings(share=[video_path, temp_path, video_format, video_frame_types, color, square, stereo, num_cameras, fps])
    frame_sync    = SyncSettings(share=[num_cameras, fps])
    tracklet_sync = SyncSettings(share=[num_cameras, fps])


class InOutGroup(Settings):
    num_artnet:         Field[int]      = Field(3, access=Field.INIT, visible=False)
    osc_sound:          OscSoundSettings
    artnets: list[ArtNetBarsSettings] = ArtNetBarsSettings(count=num_artnet)  # type: ignore[assignment]

class SmoothedFeatureGroup(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    smoother            = nodes.EuroSmootherSettings(share=[frequency])
    interpolator        = nodes.ChaseInterpolatorSettings(share=[frequency.as_('input_frequency'), output_frequency])
    prediction          = nodes.PredictorSettings(share=[frequency])

class AngleFeatureGroup(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    smoother            = nodes.EuroSmootherSettings(share=[frequency])
    angle_vel_smoother  = nodes.EuroSmootherSettings(share=[frequency])
    interpolator        = nodes.ChaseInterpolatorSettings(share=[frequency.as_('input_frequency'), output_frequency])
    prediction          = nodes.PredictorSettings(share=[frequency])

class MotionGroup(Settings):
    extractor:          nodes.AngleMotionExtractorSettings
    moving_average:     nodes.MovingAverageSettings

class PoseGroup(Settings):
    max_poses:          Field[int]          = Field(3, min=1, max=16, access=Field.INIT)
    model_type:         Field[ModelType]    = Field(ModelType.TRT, access=Field.INIT)
    model_path:         Field[str]          = Field("models", access=Field.INIT, visible=False)
    verbose:            Field[bool]         = Field(False, access=Field.INIT)
    frequency:          Field[float]        = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float]        = Field(60.0, access=Field.INIT)

    detection           = batch.DetectionSettings(share=[max_poses, model_type, model_path, verbose])
    segmentation        = batch.SegmentationSettings(share=[max_poses, model_type, model_path, verbose])
    flow                = batch.FlowSettings(share=[max_poses, model_type, model_path, verbose])
    image_crop          = batch.ImageCropSettings(share=[max_poses])
    window_similarity   = batch.WindowSimilaritySettings(share=[max_poses])
    window_correlation  = batch.WindowCorrelationSettings(share=[max_poses])
    motion_gate         = nodes.MotionGateApplicatorSettings(share=[max_poses])
    similarity_applicator = nodes.SimilarityApplicatorSettings(share=[max_poses])
    leader_applicator   = nodes.LeaderScoreApplicatorSettings(share=[max_poses])
    angle_vel_extractor = nodes.AngleVelExtractorSettings(share=[frequency.as_('fps')])
    confidence_filter:  nodes.DualConfFilterSettings
    bbox                = SmoothedFeatureGroup(share=[frequency, output_frequency])
    point               = SmoothedFeatureGroup(share=[frequency, output_frequency])
    angle               = AngleFeatureGroup(share=[frequency, output_frequency])
    similarity          = SmoothedFeatureGroup(share=[frequency, output_frequency])
    motion:             MotionGroup
    angle_sticky:       nodes.StickyFillerSettings
    similarity_sticky:  nodes.StickyFillerSettings
    angle_vel_sticky:   nodes.StickyFillerSettings
    window_raw:         trackers.WindowNodeSettings
    window_smooth:      trackers.WindowNodeSettings
    window_lerp:        trackers.WindowNodeSettings
    # rate_limiter:     nodes.RateLimiterSettings
    # easing:           nodes.EasingSettings

class TTGroup(Settings):
    timer:              TimerSettings
    tracker:            OnePerCamTrackerSettings

class MainSettings(Settings):
    num_players:        Field[int]   = Field(3, access=Field.INIT, visible=False)
    fps:                Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)

    camera              = OakGroup(share=[num_players.as_('num_cameras'), fps])
    tt:                 TTGroup
    pose                = PoseGroup(share=[num_players.as_('max_poses')])
    render:             RenderSettings
    inout               = InOutGroup(share=[num_players.as_('num_artnet')])
    server:             NiceSettings
