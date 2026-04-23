"""White Space settings — 3-camera panoramic installation with circular LED output.

PRESET MAINTENANCE
------------------
Preset JSON files live in ``files/settings/white_space/``.
Each JSON mirrors this settings tree exactly.  When you rename, add,
or remove a Field here, update every ``.json`` file in that directory
to match — delete stale keys, add new keys with their Field default.
The root class is ``Settings``.
"""

from enum import IntEnum, auto

from modules.settings import BaseSettings, NiceSettings, Field, Group, Widget
from modules.oak import CameraSettings, FrameType, CoderFormat, SimulatorSettings, RecorderSettings, SyncSettings
from modules.render import layers, ColorSettings
from modules.render.layers import DataLayerSettings
from modules.inout import OscSoundSettings, OscReceiverSettings
from modules.tracker import PanoramicTrackerSettings
from modules.pose import batch, nodes, trackers, window
from modules.pose.batch.model_types import ModelType
from modules.session import SessionSettings, SequencerSettings
from modules.gl.WindowManager import WindowSettings
from modules.WS.WSSettings import WSSettings


# ---------------------------------------------------------------------------
#  Pipeline stages & show sequencer
# ---------------------------------------------------------------------------

class Stage(IntEnum):
    RAW     = 0
    CLEAN   = auto()
    SMOOTH  = auto()
    PREDICT = auto()
    LERP    = auto()


class ShowStage(IntEnum):
    START      = 0
    PLAY_IN    = auto()
    PLAY       = auto()
    CONCLUSION = auto()
    IDLE       = auto()


class ShowSequencerSettings(SequencerSettings):
    """White Space show sequencer."""
    stages:    Field[list[ShowStage]] = Field(list(ShowStage), widget=Widget.checklist, description="Stages to play")
    durations: Field[list]            = Field([10.0, 3.0, 120.0, 10.0, 5.0], min=0.0, max=600.0, step=0.1, description="Stage durations")
    stage:     Field[ShowStage]       = Field(ShowStage.START, access=Field.READ, description="Current stage", newline=True)


# ---------------------------------------------------------------------------
#  Layers enum
# ---------------------------------------------------------------------------

class Layers(IntEnum):
    # source layers
    cam_image    = 0
    cam_mask     = auto()
    cam_crop     = auto()
    # composite
    tracker      = auto()
    poser        = auto()
    # WS visualization
    ws_tracker   = auto()
    ws_light     = auto()
    ws_lines     = auto()
    # data
    data_W       = auto()
    data_F       = auto()
    data_time    = auto()


# ---------------------------------------------------------------------------
#  Oak camera group (3 panoramic cameras)
# ---------------------------------------------------------------------------

class OakGroup(BaseSettings):
    num_cameras       : Field[int]             = Field(4, access=Field.INIT, visible=False, description="Number of cameras")
    fps               : Field[float]           = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    yolo              : Field[bool]            = Field(True, access=Field.INIT, description="Enable YOLO person detection")
    color             : Field[bool]            = Field(False, access=Field.INIT, description="Color camera (False = mono)")
    square            : Field[bool]            = Field(True, access=Field.INIT, description="Use square aspect ratio")
    stereo            : Field[bool]            = Field(False, access=Field.INIT, description="Enable stereo mode")
    hd_ready          : Field[bool]            = Field(False, access=Field.INIT, description="Use HD resolution")
    sim_enabled       : Field[bool]            = Field(False, access=Field.INIT, description="Enable simulation mode")
    model_path        : Field[str]             = Field("models", access=Field.INIT, visible=False, description="Model files directory")

    _cam_share: list = [fps, color, square, stereo, yolo, hd_ready, sim_enabled, model_path]

    cam_0     : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    cam_1     : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    cam_2     : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    cam_3     : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    simulator : Group[SimulatorSettings]         = Group(SimulatorSettings, share=[num_cameras, fps])
    tracker   : Group[PanoramicTrackerSettings]  = Group(PanoramicTrackerSettings)
    frame_sync: Group[SyncSettings]              = Group(SyncSettings, share=[num_cameras, fps])
    tracklet_sync: Group[SyncSettings]           = Group(SyncSettings, share=[num_cameras, fps])

    @property
    def cameras(self) -> list[CameraSettings]:
        return [self.cam_0, self.cam_1, self.cam_2, self.cam_3]


# ---------------------------------------------------------------------------
#  InOut group (OSC only — no ArtNet, light output via WSPipeline UDP)
# ---------------------------------------------------------------------------

class _OscSoundSettings(OscSoundSettings):
    stage: Field[Stage] = Field(Stage.LERP)


class InOutGroup(BaseSettings):
    num_players: Field[int]          = Field(8, access=Field.INIT, visible=False)
    osc_sound: Group[_OscSoundSettings] = Group(_OscSoundSettings, share=[num_players.as_('max_players')])
    osc_recv  : Group[OscReceiverSettings] = Group(OscReceiverSettings)


# ---------------------------------------------------------------------------
#  Pose feature groups  (same structure as hd_trio — keep all stages)
# ---------------------------------------------------------------------------

class BboxFeature(BaseSettings):
    frequency        : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency : Field[float] = Field(30.0)

    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])


class PointFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(30.0)

    confidence_filter: Group[nodes.DualConfFilterSettings]    = Group(nodes.DualConfFilterSettings)
    smoother         : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction       : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator     : Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])


class AngleFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(30.0)

    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky      : Group[nodes.StickyFillerSettings]      = Group(nodes.StickyFillerSettings)


class VelocityFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(30.0)

    extractor   : Group[nodes.AngleVelExtractorSettings] = Group(nodes.AngleVelExtractorSettings, share=[frequency])
    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky      : Group[nodes.StickyFillerSettings]      = Group(nodes.StickyFillerSettings)


class MotionFeature(BaseSettings):
    extractor     : Group[nodes.AngleMotionExtractorSettings] = Group(nodes.AngleMotionExtractorSettings)
    moving_average: Group[nodes.MovingAverageSettings]        = Group(nodes.MovingAverageSettings)


class SimilarityFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(30.0)
    max_poses       : Field[int]   = Field(3, min=1, max=16, access=Field.INIT)

    # pose similarity (WindowSimilarity) enabled; movement correlation disabled by default
    window_similarity    : Group[batch.WindowSimilaritySettings]      = Group(batch.WindowSimilaritySettings, share=[max_poses])
    window_correlation   : Group[batch.WindowCorrelationSettings]     = Group(batch.WindowCorrelationSettings, share=[max_poses])
    similarity_applicator: Group[nodes.SimilarityApplicatorSettings]  = Group(nodes.SimilarityApplicatorSettings, share=[max_poses])
    leader_applicator    : Group[nodes.LeaderScoreApplicatorSettings] = Group(nodes.LeaderScoreApplicatorSettings, share=[max_poses])
    smoother             : Group[nodes.EuroSmootherSettings]          = Group(nodes.EuroSmootherSettings, share=[frequency])
    interpolator         : Group[nodes.ChaseInterpolatorSettings]     = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky               : Group[nodes.StickyFillerSettings]          = Group(nodes.StickyFillerSettings)
    motion_gate          : Group[nodes.MotionGateApplicatorSettings]  = Group(nodes.MotionGateApplicatorSettings, share=[max_poses])


# ---------------------------------------------------------------------------
#  Pose pipeline group
# ---------------------------------------------------------------------------

class PoseGroup(BaseSettings):
    max_poses        : Field[int]       = Field(3, min=1, max=16, access=Field.INIT)
    model_type       : Field[ModelType] = Field(ModelType.TRT, access=Field.INIT)
    model_path       : Field[str]       = Field("models", access=Field.INIT, visible=False)
    verbose          : Field[bool]      = Field(False, access=Field.INIT)
    frequency        : Field[float]     = Field(30.0, access=Field.INIT)
    output_frequency : Field[float]     = Field(30.0)
    use_segmentation : Field[bool]      = Field(False, access=Field.INIT, description="Enable mask/segmentation extraction")
    ws_input_stage   : Field[Stage]     = Field(Stage.LERP, description="Pipeline stage that feeds the WS light pipeline")

    _feature_share: list = [frequency, output_frequency]

    detection       : Group[batch.DetectionSettings]          = Group(batch.DetectionSettings, share=[max_poses, model_type, model_path, verbose])
    segmentation    : Group[batch.SegmentationSettings]       = Group(batch.SegmentationSettings, share=[max_poses, model_type, model_path, verbose, use_segmentation.as_('enabled')])
    image_crop      : Group[batch.ImageCropSettings]          = Group(batch.ImageCropSettings, share=[max_poses])
    angle_extractor : Group[nodes.AngleExtractorSettings]     = Group(nodes.AngleExtractorSettings)
    bbox            : Group[BboxFeature]                      = Group(BboxFeature, share=_feature_share)
    point           : Group[PointFeature]                     = Group(PointFeature, share=_feature_share)
    angle           : Group[AngleFeature]                     = Group(AngleFeature, share=_feature_share)
    velocity        : Group[VelocityFeature]                  = Group(VelocityFeature, share=_feature_share)
    motion          : Group[MotionFeature]                    = Group(MotionFeature)
    similarity      : Group[SimilarityFeature]                = Group(SimilarityFeature, share=[frequency, output_frequency, max_poses])
    window_raw      : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_clean    : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_smooth   : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_predict  : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_lerp     : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)


# ---------------------------------------------------------------------------
#  Session group (recording lifecycle)
# ---------------------------------------------------------------------------

class SessionGroup(BaseSettings):
    """App session group — composes SessionSettings with app-specific recorders."""
    num_cameras:   Field[int]   = Field(4, access=Field.INIT, description="Number of cameras")
    fps:           Field[float] = Field(30.0, access=Field.INIT, description="Camera frame rate")

    start:         Field[bool]  = Field(False, widget=Widget.button, description="Start session")
    stop:          Field[bool]  = Field(False, widget=Widget.button, description="Stop session")
    output_path:   Field[str]   = Field("recordings", access=Field.INIT, description="Recordings output directory")
    name:          Field[str]   = Field("", widget=Widget.input, description="Recording group ID")
    split:         Field[bool]  = Field(False, widget=Widget.button, description="Split chunk", visible=False)
    split_seconds: Field[float] = Field(10, min=1, max=60, widget=Widget.number, description="Split recording into chunks (seconds)")

    _session_share:  list = [output_path, name, start, stop, split, split_seconds]
    _recorder_share: list = [start, stop, split, name, output_path]

    osc:      Group[OscReceiverSettings]    = Group(OscReceiverSettings)
    core:     Group[SessionSettings]        = Group(SessionSettings, share=_session_share)
    sequencer: Group[ShowSequencerSettings] = Group(ShowSequencerSettings, share=[start, stop])
    video:    Group[RecorderSettings]       = Group(RecorderSettings, share=_recorder_share + [num_cameras, fps])


# ---------------------------------------------------------------------------
#  Render settings
# ---------------------------------------------------------------------------

class _TrackerCompSettings(layers.TrackerCompSettings):
    stage: Field[Stage] = Field(Stage.LERP)

class _PoseCompSettings(layers.PoseCompSettings):
    stage: Field[Stage] = Field(Stage.LERP)


class PreviewGroup(BaseSettings):
    tracker: Group[_TrackerCompSettings] = Group(_TrackerCompSettings)
    poser  : Group[_PoseCompSettings]    = Group(_PoseCompSettings)


class RenderSettings(BaseSettings):
    num_cams:    Field[int]  = Field(4, access=Field.INIT, visible=False, description="Number of cameras")
    num_players: Field[int]  = Field(4, access=Field.INIT, visible=False, description="Number of players")
    preview:     Group[PreviewGroup]     = Group(PreviewGroup)
    data:        Group[DataLayerSettings] = Group(DataLayerSettings)
    colors:      Group[ColorSettings]    = Group(ColorSettings)
    window:      Group[WindowSettings]   = Group(WindowSettings)


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    num_players: Field[int]   = Field(8, access=Field.INIT)
    num_cameras: Field[int]   = Field(4, access=Field.INIT)
    input_fps  : Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)
    render_fps : Field[float] = Field(30.0)

    camera : Group[OakGroup]        = Group(OakGroup, share=[num_cameras.as_('num_cameras'), input_fps.as_('fps')])
    inout  : Group[InOutGroup]      = Group(InOutGroup, share=[num_players.as_('num_players')])
    pose   : Group[PoseGroup]       = Group(PoseGroup, share=[num_players.as_('max_poses'), input_fps.as_('frequency'), render_fps.as_('output_frequency')])
    ws     : Group[WSSettings]      = Group(WSSettings, share=[num_players.as_('max_poses'), input_fps.as_('light_rate')])
    render : Group[RenderSettings]  = Group(RenderSettings, share=[num_players, num_cameras.as_('num_cams')])
    server : Group[NiceSettings]    = Group(NiceSettings)
    session: Group[SessionGroup]    = Group(SessionGroup, share=[num_cameras.as_('num_cameras'), input_fps.as_('fps')])
