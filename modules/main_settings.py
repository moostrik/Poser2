# TODO
# log visible in gui
# what to do with mainsettings in tabs
# share fps / frequencies across all settings
# better width handling in nice panel

from modules.settings import Settings, NiceSettings, Field, Group
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

    _cam_share = [fps, color, square, stereo, yolo, hd_ready, sim_enabled, model_path]
    cam_0       = Group(CameraSettings, share=_cam_share)
    cam_1       = Group(CameraSettings, share=_cam_share)
    cam_2       = Group(CameraSettings, share=_cam_share)
    simulator   = Group(SimulatorSettings, share=[video_path, video_format, video_frame_types, num_cameras, color, square, stereo])
    recorder    = Group(RecorderSettings, share=[video_path, temp_path, video_format, video_frame_types, color, square, stereo, num_cameras, fps])
    frame_sync  = Group(SyncSettings, share=[num_cameras, fps])
    tracklet_sync = Group(SyncSettings, share=[num_cameras, fps])

    @property
    def cameras(self) -> list[CameraSettings]:
        return [self.cam_0, self.cam_1, self.cam_2]


class InOutGroup(Settings):
    osc_sound   = Group(OscSoundSettings)
    artnet_0    = Group(ArtNetBarsSettings)
    artnet_1    = Group(ArtNetBarsSettings)
    artnet_2    = Group(ArtNetBarsSettings)

    @property
    def artnets(self) -> list[ArtNetBarsSettings]:
        return [self.artnet_0, self.artnet_1, self.artnet_2]

class BboxFeature(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    smoother     = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction   = Group(nodes.PredictorSettings, share=[frequency])
    interpolator = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class PointFeature(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    confidence_filter = Group(nodes.DualConfFilterSettings)
    smoother     = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction   = Group(nodes.PredictorSettings, share=[frequency])
    interpolator = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class AngleFeature(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    smoother     = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction   = Group(nodes.PredictorSettings, share=[frequency])
    interpolator = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky       = Group(nodes.StickyFillerSettings)

class VelocityFeature(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    extractor     = Group(nodes.AngleVelExtractorSettings, share=[frequency])
    smoother         = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction          = Group(nodes.PredictorSettings, share=[frequency])
    interpolator  = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky           = Group(nodes.StickyFillerSettings)

class MotionFeature(Settings):
    extractor       = Group(nodes.AngleMotionExtractorSettings)
    moving_average  = Group(nodes.MovingAverageSettings)

class SimilarityFeature(Settings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)
    max_poses:          Field[int]   = Field(3, min=1, max=16, access=Field.INIT)

    window_similarity     = Group(batch.WindowSimilaritySettings, share=[max_poses])
    window_correlation    = Group(batch.WindowCorrelationSettings, share=[max_poses])
    similarity_applicator = Group(nodes.SimilarityApplicatorSettings, share=[max_poses])
    leader_applicator     = Group(nodes.LeaderScoreApplicatorSettings, share=[max_poses])
    smoother              = Group(nodes.EuroSmootherSettings, share=[frequency])
    interpolator          = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky                = Group(nodes.StickyFillerSettings)
    motion_gate           = Group(nodes.MotionGateApplicatorSettings, share=[max_poses])

class PoseGroup(Settings):
    max_poses:          Field[int]          = Field(3, min=1, max=16, access=Field.INIT)
    model_type:         Field[ModelType]    = Field(ModelType.TRT, access=Field.INIT)
    model_path:         Field[str]          = Field("models", access=Field.INIT, visible=False)
    verbose:            Field[bool]         = Field(False, access=Field.INIT)
    frequency:          Field[float]        = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float]        = Field(60.0, access=Field.INIT)

    _batch_share = [max_poses, model_type, model_path, verbose]
    _feature_share = [frequency, output_frequency]

    detection    = Group(batch.DetectionSettings, share=_batch_share)
    segmentation = Group(batch.SegmentationSettings, share=_batch_share)
    flow         = Group(batch.FlowSettings, share=_batch_share)
    image_crop   = Group(batch.ImageCropSettings, share=[max_poses])
    bbox         = Group(BboxFeature, share=_feature_share)
    point        = Group(PointFeature, share=_feature_share)
    angle        = Group(AngleFeature, share=_feature_share)
    velocity     = Group(VelocityFeature, share=_feature_share)
    motion       = Group(MotionFeature)
    similarity   = Group(SimilarityFeature, share=[frequency, output_frequency, max_poses])
    window_raw   = Group(trackers.WindowNodeSettings)
    window_smooth = Group(trackers.WindowNodeSettings)
    window_lerp  = Group(trackers.WindowNodeSettings)

class TTGroup(Settings):
    timer   = Group(TimerSettings)
    tracker = Group(OnePerCamTrackerSettings)

class MainSettings(Settings):
    num_players:        Field[int]   = Field(3, access=Field.INIT, visible=False)
    fps:                Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)

    camera  = Group(OakGroup, share=[num_players.as_('num_cameras'), fps])
    tt      = Group(TTGroup)
    pose    = Group(PoseGroup, share=[num_players.as_('max_poses'), fps.as_('frequency')])
    render  = Group(RenderSettings)
    inout   = Group(InOutGroup)
    server  = Group(NiceSettings)
