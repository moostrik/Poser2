"""HD Trio settings — 3-camera interactive installation with fluid rendering.

PRESET MAINTENANCE
------------------
Preset JSON files live in ``files/settings/hd_trio/``.
Each JSON mirrors this settings tree exactly.  When you rename, add,
or remove a Field here, update every ``.json`` file in that directory
to match — delete stale keys, add new keys with their Field default.
The root class is ``HDTrioSettings``.
"""

from enum import IntEnum, auto
from typing import Any

from modules.settings import BaseSettings, NiceSettings, Field, Group, Widget
from modules.oak import CameraSettings, FrameType, CoderFormat, SimulatorSettings, RecorderSettings, SyncSettings
from modules.render.color_settings import ColorSettings
from modules.render import layers
from modules.inout import OscSoundSettings, ArtNetBarsSettings, OscReceiverSettings
from modules.tracker import OnePerCamTrackerSettings
from modules.pose import batch, nodes, trackers
from modules.pose.batch.model_types import ModelType
from modules.pose.recorder.settings import RecorderSettings as PoseRecorderSettings
from modules.session import SessionSettings, TimelineSettings
from modules.gl.WindowManager import WindowSettings


# ---------------------------------------------------------------------------
#  Show stages & timeline settings
# ---------------------------------------------------------------------------

class ShowStage(IntEnum):
    START =         0
    INTRODUCTION =  auto()
    PLAY =          auto()
    CONCLUSION =    auto()
    COOLDOWN =      auto()
    FREEROAM =      auto()


class ShowTimelineSettings(TimelineSettings):
    """HD Trio show timeline with project-specific stages."""
    stages:     Field[list[ShowStage]] = Field(list(ShowStage), widget=Widget.checklist, description="Stages to play")
    durations:  Field[list] = Field([3.0, 6.0, 60.0, 6.0, 3.0], min=0.0, max=600.0, step=0.1, description="Stage durations")
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
    color             : Field[bool]            = Field(True, access=Field.INIT, description="Enable color capture")
    mono              : Field[bool]            = Field(False, access=Field.INIT, description="Enable mono capture")
    square            : Field[bool]            = Field(True, access=Field.INIT, description="Use square aspect ratio")
    stereo            : Field[bool]            = Field(False, access=Field.INIT, description="Enable stereo mode")
    hd_ready          : Field[bool]            = Field(False, access=Field.INIT, description="Use HD resolution")
    sim_enabled       : Field[bool]            = Field(False, access=Field.INIT, description="Enable simulation mode")
    model_path        : Field[str]             = Field("models", access=Field.INIT, visible=False, description="Model files directory")

    _cam_share: list = [fps, color, mono, square, stereo, yolo, hd_ready, sim_enabled, model_path]

    cam_0        : Group[CameraSettings]    = Group(CameraSettings, share=_cam_share)
    cam_1        : Group[CameraSettings]    = Group(CameraSettings, share=_cam_share)
    cam_2        : Group[CameraSettings]    = Group(CameraSettings, share=_cam_share)
    simulator    : Group[SimulatorSettings]    = Group(SimulatorSettings, share=[num_cameras, fps])
    tracker      : Group[OnePerCamTrackerSettings] = Group(OnePerCamTrackerSettings)
    frame_sync   : Group[SyncSettings]         = Group(SyncSettings, share=[num_cameras, fps])
    tracklet_sync: Group[SyncSettings]         = Group(SyncSettings, share=[num_cameras, fps])

    @property
    def cameras(self) -> list[CameraSettings]:
        return [self.cam_0, self.cam_1, self.cam_2]


# ---------------------------------------------------------------------------
#  InOut group (OSC + 3 ArtNet controllers)
# ---------------------------------------------------------------------------

class InOutGroup(BaseSettings):
    osc_sound:        Group[OscSoundSettings]    = Group(OscSoundSettings)
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
    max_poses       : Field[int]       = Field(3, min=1, max=16, access=Field.INIT)
    model_type      : Field[ModelType] = Field(ModelType.TRT, access=Field.INIT)
    model_path      : Field[str]       = Field("models", access=Field.INIT, visible=False)
    verbose         : Field[bool]      = Field(False, access=Field.INIT)
    frequency       : Field[float]     = Field(30.0, access=Field.INIT)
    output_frequency: Field[float]     = Field(60.0)

    _batch_share  : list = [max_poses, model_type, model_path, verbose]
    _feature_share: list = [frequency, output_frequency]

    detection    : Group[batch.DetectionSettings]    = Group(batch.DetectionSettings, share=_batch_share)
    segmentation : Group[batch.SegmentationSettings] = Group(batch.SegmentationSettings, share=_batch_share)
    flow         : Group[batch.FlowSettings]         = Group(batch.FlowSettings, share=_batch_share)
    image_crop   : Group[batch.ImageCropSettings]    = Group(batch.ImageCropSettings, share=[max_poses])
    bbox         : Group[BboxFeature]                = Group(BboxFeature, share=_feature_share)
    point        : Group[PointFeature]               = Group(PointFeature, share=_feature_share)
    angle        : Group[AngleFeature]               = Group(AngleFeature, share=_feature_share)
    velocity     : Group[VelocityFeature]            = Group(VelocityFeature, share=_feature_share)
    motion       : Group[MotionFeature]              = Group(MotionFeature)
    similarity   : Group[SimilarityFeature]          = Group(SimilarityFeature, share=[frequency, output_frequency, max_poses])
    window_raw   : Group[nodes.WindowNodeSettings]   = Group(trackers.WindowNodeSettings)
    window_smooth: Group[nodes.WindowNodeSettings]   = Group(trackers.WindowNodeSettings)
    window_lerp  : Group[nodes.WindowNodeSettings]   = Group(trackers.WindowNodeSettings)


# ---------------------------------------------------------------------------
#  Session group (recording lifecycle: OSC, timer, video & pose recorders)
# ---------------------------------------------------------------------------

class SessionGroup(SessionSettings):
    num_cameras: Field[int]   = Field(1, access=Field.INIT, description="Number of cameras")
    fps:         Field[float] = Field(30.0, access=Field.INIT, description="Camera frame rate")

    _recorder_share: list = [SessionSettings.record, SessionSettings.split, SessionSettings.group_id.as_('name'), SessionSettings.output_path]

    osc     : Group[OscReceiverSettings]     = Group(OscReceiverSettings)
    timeline: Group[ShowTimelineSettings]    = Group(ShowTimelineSettings) #, share=[SessionSettings.record.as_('run')])
    video   : Group[RecorderSettings]        = Group(RecorderSettings, share=_recorder_share + [num_cameras, fps])
    pose    : Group[PoseRecorderSettings]    = Group(PoseRecorderSettings, share=_recorder_share)


# ---------------------------------------------------------------------------
#  Render settings (layer configs, centre, flow, fluid, colors, window)
# ---------------------------------------------------------------------------

class LayerGroup(BaseSettings):
    select: Group[LayerSettings]                 = Group(LayerSettings)
    lut   : Group[layers.CompositeLayerSettings] = Group(layers.CompositeLayerSettings)

class DataGroup(BaseSettings):
    a: Group[layers.DataLayerSettings] = Group(layers.DataLayerSettings)
    b: Group[layers.DataLayerSettings] = Group(layers.DataLayerSettings)

class PreviewGroup(BaseSettings):
    tracker: Group[layers.TrackerCompSettings] = Group(layers.TrackerCompSettings)
    poser  : Group[layers.PoseCompSettings]    = Group(layers.PoseCompSettings)

class CentreGroup(BaseSettings):
    geometry: Group[layers.CentreGeomSettings]     = Group(layers.CentreGeomSettings)
    mask    : Group[layers.CentreMaskSettings]     = Group(layers.CentreMaskSettings)
    cam     : Group[layers.CentreCamSettings]      = Group(layers.CentreCamSettings)
    frg     : Group[layers.CentreFrgSettings]      = Group(layers.CentreFrgSettings)
    pose    : Group[layers.CentrePoseSettings]     = Group(layers.CentrePoseSettings)
    color   : Group[layers.ColorMaskLayerSettings] = Group(layers.ColorMaskLayerSettings)

class RenderGroup(BaseSettings):
    layer  : Group[LayerGroup]                = Group(LayerGroup)
    data   : Group[DataGroup]                 = Group(DataGroup)
    preview: Group[PreviewGroup]              = Group(PreviewGroup)
    centre : Group[CentreGroup]               = Group(CentreGroup)
    flow   : Group[layers.FlowLayerSettings]  = Group(layers.FlowLayerSettings)
    fluid  : Group[layers.FluidLayerSettings] = Group(layers.FluidLayerSettings)
    colors : Group[ColorSettings]             = Group(ColorSettings)
    window : Group[WindowSettings]            = Group(WindowSettings)


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class HDTrioSettings(BaseSettings):
    num_players: Field[int]   = Field(3, access=Field.INIT)
    input_fps  : Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)
    render_fps : Field[float] = Field(60.0)

    camera : Group[OakGroup]     = Group(OakGroup, share=[num_players.as_('num_cameras'), input_fps.as_('fps')])
    inout  : Group[InOutGroup]   = Group(InOutGroup)
    pose   : Group[PoseGroup]    = Group(PoseGroup, share=[num_players.as_('max_poses'), input_fps.as_('frequency'), render_fps.as_('output_frequency')])
    render : Group[RenderGroup]  = Group(RenderGroup)
    server : Group[NiceSettings] = Group(NiceSettings)
    session: Group[SessionGroup] = Group(SessionGroup, share=[num_players.as_('num_cameras'), input_fps.as_('fps')])
