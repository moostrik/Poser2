"""HD Trio settings — 3-camera interactive installation with fluid rendering.

PRESET MAINTENANCE
------------------
Preset JSON files live in ``files/settings/hd_trio/``.
Each JSON mirrors this settings tree exactly.  When you rename, add,
or remove a Field here, update every ``.json`` file in that directory
to match — delete stale keys, add new keys with their Field default.
The root class is ``Settings``.
"""

from enum import IntEnum, auto
from typing import Any

from modules.settings import BaseSettings, NiceSettings, Field, Group, Widget
from modules.oak import CameraSettings, FrameType, CoderFormat, SimulatorSettings, RecorderSettings, SyncSettings
from modules.render.color_settings import ColorSettings
from modules.render import layers
from modules.render.layers.centre.CentrePoseLayer import CentrePoseSettings
from modules.inout import OscSoundSettings, ArtNetBarsSettings, OscReceiverSettings
from modules.utils import Color
from modules.tracker import OnePerCamTrackerSettings
from modules.pose import nodes, trackers, window, analytics
from modules import inference
from modules.inference import ModelType
from modules.pose.recorder.settings import RecorderSettings as PoseRecorderSettings
from modules.session import SessionSettings, SequencerSettings
from modules.gl.WindowManager import WindowSettings


# ---------------------------------------------------------------------------
#  Show stages & sequencer settings
# ---------------------------------------------------------------------------

class Stage(IntEnum):
    RAW =       0
    CLEAN =     auto()
    SMOOTH =    auto()
    PREDICT =   auto()
    LERP =      auto()


class ShowStage(IntEnum):
    START =         0
    INTRO_IN =      auto()
    INTRO =         auto()
    INTRO_OUT =     auto()
    PLAY_IN =       auto()
    PLAY =          auto()
    CONCLUSION =    auto()
    IDLE =          auto()


class ShowSequencerSettings(SequencerSettings):
    """HD Trio show sequencer with project-specific stages."""
    stages:     Field[list[ShowStage]] = Field(list(ShowStage), widget=Widget.checklist, description="Stages to play")
    durations:  Field[list] = Field([10.0, 3.0, 30.0, 3.0, 3.0, 60.0, 10.0, 3.0], min=0.0, max=600.0, step=0.1, description="Stage durations")
    stage:      Field[ShowStage]   = Field(ShowStage.START, access=Field.READ, description="Current stage", newline=True)


# ---------------------------------------------------------------------------
#  Layers enum  (defines which render layers this app uses)
# ---------------------------------------------------------------------------

class Layers(IntEnum):
    # source layers
    cam_image =     0
    cam_mask =      auto()
    cam_frg=        auto()
    cam_crop =      auto()
    # composite layers
    tracker =       auto()
    poser =         auto()
    # centre layers
    centre_geom=    auto()
    centre_cam =    auto()
    centre_mask =   auto()
    centre_frg =    auto()
    centre_pose =   auto()
    # intro overlay
    intro_pose =    auto()
    # data layers
    data_B_W =      auto()
    data_B_F =      auto()
    data_A_W =      auto()
    data_A_F =      auto()
    data_time =     auto()
    # composition layers
    color_mask =    auto()
    flow =          auto()
    fluid =         auto()
    composite =     auto()


# ---------------------------------------------------------------------------
#  Layer selection settings
# ---------------------------------------------------------------------------

class LayerSettings(BaseSettings):
    preview: Field[list[Layers]] = Field([Layers.composite], description="Layers drawn in the preview viewports", widget=Widget.playlist)
    final:   Field[list[Layers]] = Field([Layers.composite], description="Layers drawn on the output monitors", widget=Widget.playlist)


# ---------------------------------------------------------------------------
#  Oak camera group (3 cameras)
# ---------------------------------------------------------------------------

class OakGroup(BaseSettings):
    num_cameras       : Field[int]             = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    fps               : Field[float]           = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    yolo              : Field[bool]            = Field(True, access=Field.INIT, description="Enable YOLO person detection")
    color             : Field[bool]            = Field(True, access=Field.INIT, description="Color camera (False = mono)")
    square            : Field[bool]            = Field(True, access=Field.INIT, description="Use square aspect ratio")
    stereo            : Field[bool]            = Field(False, access=Field.INIT, description="Enable stereo mode")
    hd_ready          : Field[bool]            = Field(False, access=Field.INIT, description="Use HD resolution")
    sim_enabled       : Field[bool]            = Field(False, access=Field.INIT, description="Enable simulation mode")
    model_path        : Field[str]             = Field("models", access=Field.INIT, visible=False, description="Model files directory")

    _cam_share: list = [fps, color, square, stereo, yolo, hd_ready, sim_enabled, model_path]

    cam_0        : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    cam_1        : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    cam_2        : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    simulator    : Group[SimulatorSettings]         = Group(SimulatorSettings, share=[num_cameras, fps])
    tracker      : Group[OnePerCamTrackerSettings]  = Group(OnePerCamTrackerSettings)
    frame_sync   : Group[SyncSettings]              = Group(SyncSettings, share=[num_cameras, fps])
    tracklet_sync: Group[SyncSettings]              = Group(SyncSettings, share=[num_cameras, fps])

    @property
    def cameras(self) -> list[CameraSettings]:
        return [self.cam_0, self.cam_1, self.cam_2]


# ---------------------------------------------------------------------------
#  InOut group (OSC + 3 ArtNet controllers)
# ---------------------------------------------------------------------------

class _OscSoundSettings(OscSoundSettings):
    stage: Field[Stage] = Field(Stage.LERP)


class InOutGroup(BaseSettings):
    osc_sound:        Group[_OscSoundSettings]   = Group(_OscSoundSettings)
    artnet_0 :        Group[ArtNetBarsSettings]  = Group(ArtNetBarsSettings)
    artnet_1 :        Group[ArtNetBarsSettings]  = Group(ArtNetBarsSettings)
    artnet_2 :        Group[ArtNetBarsSettings]  = Group(ArtNetBarsSettings)

    @property
    def artnets(self) -> list[ArtNetBarsSettings]:
        return [self.artnet_0, self.artnet_1, self.artnet_2]


# ---------------------------------------------------------------------------
#  Pose feature groups
# ---------------------------------------------------------------------------

class BboxFeature(BaseSettings):
    frequency        : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency : Field[float] = Field(60.0)

    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class PointFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(60.0)

    confidence_filter: Group[nodes.DualConfFilterSettings]    = Group(nodes.DualConfFilterSettings)
    smoother         : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction       : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator     : Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class AngleFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(60.0)

    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky      : Group[nodes.StickyFillerSettings]      = Group(nodes.StickyFillerSettings)

class VelocityFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(60.0)

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
    output_frequency: Field[float] = Field(60.0)
    max_poses       : Field[int]   = Field(3, min=1, max=16, access=Field.INIT)

    window_similarity    : Group[analytics.WindowSimilaritySettings]      = Group(analytics.WindowSimilaritySettings, share=[max_poses])
    window_correlation   : Group[analytics.WindowCorrelationSettings]     = Group(analytics.WindowCorrelationSettings, share=[max_poses])
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
    max_poses       : Field[int]       = Field(3, min=1, max=16, access=Field.INIT)
    model_type      : Field[ModelType] = Field(ModelType.TRT, access=Field.INIT)
    model_path      : Field[str]       = Field("models", access=Field.INIT, visible=False)
    verbose         : Field[bool]      = Field(False, access=Field.INIT)
    frequency       : Field[float]     = Field(30.0, access=Field.INIT)
    output_frequency: Field[float]     = Field(60.0)

    _batch_share  : list = [max_poses, model_type, model_path, verbose]
    _feature_share: list = [frequency, output_frequency]

    detection       : Group[inference.DetectionSettings]          = Group(inference.DetectionSettings, share=_batch_share)
    segmentation    : Group[inference.SegmentationSettings]       = Group(inference.SegmentationSettings, share=_batch_share)
    flow            : Group[inference.FlowSettings]               = Group(inference.FlowSettings, share=_batch_share)
    image_crop      : Group[inference.CropSettings]          = Group(inference.CropSettings, share=[max_poses])
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


class _PoseRecorderSettings(PoseRecorderSettings):
    stage: Field[Stage] = Field(Stage.RAW)


# ---------------------------------------------------------------------------
#  Session group (recording lifecycle: OSC, timer, video & pose recorders)
# ---------------------------------------------------------------------------

class SessionGroup(BaseSettings):
    """App session group — composes SessionSettings with app-specific recorders."""
    num_cameras:   Field[int]   = Field(1, access=Field.INIT, description="Number of cameras")
    fps:           Field[float] = Field(30.0, access=Field.INIT, description="Camera frame rate")

    start:         Field[bool]  = Field(False, widget=Widget.button, description="Start session")
    stop:          Field[bool]  = Field(False, widget=Widget.button, description="Stop session")
    output_path:   Field[str]   = Field("recordings", access=Field.INIT, description="Recordings output directory")
    name:          Field[str]   = Field("", widget=Widget.input, description="Recording group ID")
    split:         Field[bool]  = Field(False, widget=Widget.button, description="Split chunk", visible=False)
    split_seconds: Field[float] = Field(10, min=1, max=60, widget=Widget.number, description="Split recording into chunks of this length (seconds)")

    _session_share:  list = [output_path, name, start, stop, split, split_seconds]
    _recorder_share: list = [start, stop, split, name, output_path]

    osc     : Group[OscReceiverSettings]     = Group(OscReceiverSettings)
    core    : Group[SessionSettings]         = Group(SessionSettings, share=_session_share)
    sequencer: Group[ShowSequencerSettings]   = Group(ShowSequencerSettings, share=[start, stop])
    video   : Group[RecorderSettings]        = Group(RecorderSettings, share=_recorder_share + [num_cameras, fps])
    pose    : Group[_PoseRecorderSettings]   = Group(_PoseRecorderSettings, share=_recorder_share)


# ---------------------------------------------------------------------------
#  Render settings (layer configs, centre, flow, fluid, colors, window)
# ---------------------------------------------------------------------------

# Stage-aware subclasses — override Field[int] with Field[Stage] for dropdown UI
class _DataLayerSettings(layers.DataLayerSettings):
    stage: Field[Stage] = Field(Stage.SMOOTH)

class _TrackerCompSettings(layers.TrackerCompSettings):
    stage: Field[Stage] = Field(Stage.LERP)

class _PoseCompSettings(layers.PoseCompSettings):
    stage: Field[Stage] = Field(Stage.LERP)

class _CentreGeomSettings(layers.CentreGeomSettings):
    stage: Field[Stage] = Field(Stage.SMOOTH)


class LayerGroup(BaseSettings):
    select: Group[LayerSettings]                 = Group(LayerSettings)
    lut   : Group[layers.CompositeLayerSettings] = Group(layers.CompositeLayerSettings)

class DataGroup(BaseSettings):
    a: Group[_DataLayerSettings] = Group(_DataLayerSettings)
    b: Group[_DataLayerSettings] = Group(_DataLayerSettings)

class PreviewGroup(BaseSettings):
    tracker: Group[_TrackerCompSettings] = Group(_TrackerCompSettings)
    poser  : Group[_PoseCompSettings]    = Group(_PoseCompSettings)

class CentreGroup(BaseSettings):
    geometry: Group[_CentreGeomSettings]            = Group(_CentreGeomSettings)
    mask    : Group[layers.CentreMaskSettings]      = Group(layers.CentreMaskSettings)
    cam     : Group[layers.CentreCamSettings]       = Group(layers.CentreCamSettings)
    frg     : Group[layers.CentreFrgSettings]       = Group(layers.CentreFrgSettings)
    pose    : Group[layers.CentrePoseSettings]      = Group(layers.CentrePoseSettings)
    color   : Group[layers.ColorMaskLayerSettings]  = Group(layers.ColorMaskLayerSettings)

class IntroSequenceSettings(BaseSettings):
    """Settings for prerecorded pose overlay during INTRO stages."""
    verbose:        Field[bool]  = Field(False, description="Log start/stop events")
    recording_path: Field[str]   = Field("recordings/intro", access=Field.INIT, description="Path to recorded pose folder")
    source_track:   Field[int]   = Field(0, min=0, max=16, access=Field.INIT, description="Track ID to use from recording")
    color:          Field[Color] = Field(Color(1.0, 1.0, 1.0), description="Skeleton color for intro overlay")
    pose:           Group[CentrePoseSettings]  = Group(CentrePoseSettings)


class RenderSettings(BaseSettings):
    stage:       Field[Stage] = Field(Stage.LERP, description="Pipeline stage for flow/fluid/color layers")
    num_cams:    Field[int]   = Field(3, access=Field.INIT, visible=False, description="Number of cameras")
    num_players: Field[int]   = Field(3, access=Field.INIT, visible=False, description="Number of players")
    layer  : Group[LayerGroup]                = Group(LayerGroup)
    data   : Group[DataGroup]                 = Group(DataGroup)
    preview: Group[PreviewGroup]              = Group(PreviewGroup)
    centre : Group[CentreGroup]               = Group(CentreGroup)
    intro_sequence: Group[IntroSequenceSettings] = Group(IntroSequenceSettings)
    flow   : Group[layers.FlowLayerSettings]  = Group(layers.FlowLayerSettings, share=[stage])
    fluid  : Group[layers.FluidLayerSettings] = Group(layers.FluidLayerSettings, share=[stage])
    colors : Group[ColorSettings]             = Group(ColorSettings)
    window : Group[WindowSettings]            = Group(WindowSettings)


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    num_players: Field[int]   = Field(3, access=Field.INIT)
    input_fps  : Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)
    render_fps : Field[float] = Field(60.0)

    camera : Group[OakGroup]     = Group(OakGroup, share=[num_players.as_('num_cameras'), input_fps.as_('fps')])
    inout  : Group[InOutGroup]   = Group(InOutGroup)
    pose   : Group[PoseGroup]    = Group(PoseGroup, share=[num_players.as_('max_poses'), input_fps.as_('frequency'), render_fps.as_('output_frequency')])
    render : Group[RenderSettings]  = Group(RenderSettings, share=[num_players, num_players.as_('num_cams')])
    server : Group[NiceSettings] = Group(NiceSettings)
    session: Group[SessionGroup] = Group(SessionGroup, share=[num_players.as_('num_cameras'), input_fps.as_('fps')])
