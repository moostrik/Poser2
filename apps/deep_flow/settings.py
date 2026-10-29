"""Deep Flow settings — single-camera 3D volumetric fluid installation.

PRESET MAINTENANCE
------------------
Preset JSON files live in ``apps/deep_flow/data/settings/``.
Each JSON mirrors this settings tree exactly.  When you rename, add,
or remove a Field here, update every ``.json`` file in that directory
to match — delete stale keys, add new keys with their Field default.
The root class is ``Settings``.
"""

from enum import IntEnum, auto

from modules.settings import BaseSettings, Field, Group, Widget
from modules.panel import PanelSettings
from modules.camera import CameraCheckSettings, CameraResolution, PlayerSettings, RecorderSettings, SyncSettings
from modules.camera.oak import OakCameraSettings
from modules.render import layers, ColorSettings
from modules.inout import OscSoundSettings, OscReceiverSettings
from modules.tracker import OnePerCamTrackerSettings, PosesFromTrackletsSettings
from modules.pose import nodes, trackers, window
from modules import inference
from modules.gl import WindowSettings


# ---------------------------------------------------------------------------
#  Pipeline stages
# ---------------------------------------------------------------------------

class Stage(IntEnum):
    RAW =       0
    CLEAN =     auto()
    SMOOTH =    auto()
    PREDICT =   auto()
    LERP =      auto()


# ---------------------------------------------------------------------------
#  Layers enum
# ---------------------------------------------------------------------------

class Layers(IntEnum):
    # source layers
    cam_image =     0
    cam_mask =      auto()
    cam_foreground = auto()
    cam_crop =      auto()
    # composite layers
    tracker =       auto()
    pose =          auto()
    cam_readings =  auto()   # the camera's frame rate, tilt and roll error; draw it over tracker
    # center layers
    center_geometry = auto()
    center_cam =    auto()
    center_mask =   auto()
    center_foreground = auto()
    center_pose =   auto()
    # data layers
    data_B_W =      auto()
    data_B_F =      auto()
    data_A_W =      auto()
    data_A_F =      auto()
    data_time =     auto()
    # composition layers
    color_mask =    auto()
    flow =          auto()
    fluid3d =       auto()
    composite =     auto()


# ---------------------------------------------------------------------------
#  Layer selection settings (split view)
# ---------------------------------------------------------------------------

class LayerSettings(BaseSettings):
    left:      Field[list[Layers]] = Field([Layers.composite],                    description="Layers drawn in the left viewport",       widget=Widget.playlist)
    right:     Field[list[Layers]] = Field([Layers.pose],                         description="Layers drawn in the right viewport",      widget=Widget.playlist)
    composite: Field[list[Layers]] = Field([Layers.fluid3d, Layers.color_mask],   description="Layers composited into composite output", widget=Widget.playlist)
    final:     Field[list[Layers]] = Field([Layers.composite],                    description="Layers drawn on the output monitors",     widget=Widget.playlist)


# ---------------------------------------------------------------------------
#  Oak camera group (1 camera)
# ---------------------------------------------------------------------------

class OakGroup(BaseSettings):
    num_cams:        Field[int]               = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    fps:                Field[float]             = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    keyframe_interval:  Field[int]               = Field(1, access=Field.INIT, visible=False, description="Keyframe interval in frames (1 = every frame)")
    yolo:               Field[bool]              = Field(True, access=Field.INIT, description="Enable YOLO person detection")
    color:              Field[bool]              = Field(True, access=Field.INIT, description="Enable color capture")
    square:             Field[bool]              = Field(True, access=Field.INIT, description="Use square aspect ratio")
    resolution:         Field[CameraResolution]  = Field(CameraResolution.P1080, access=Field.INIT, description="Sensor mode. P720 and P800 on both sensors, P1080 color only.")
    sim_enabled:        Field[bool]              = Field(False, access=Field.INIT, description="Enable simulation mode")
    model_path:         Field[str]               = Field("data/models", access=Field.INIT, visible=False, description="Model files directory")
    fov:                Field[float]             = Field(127.0, access=Field.INIT, description="Camera horizontal FOV (°) — OAK-D Pro W, OV9782 color, 1280x800 native. 720p is a vertical crop, so the full width and this figure both hold.")

    _cam_share: list = [fps, color, square, yolo, resolution, model_path, fov]
    cam_0          : Group[OakCameraSettings]              = Group(OakCameraSettings, share=_cam_share)
    camera_check   : Group[CameraCheckSettings]            = Group(CameraCheckSettings)
    simulator      : Group[PlayerSettings]                 = Group(PlayerSettings, share=[num_cams, fps])
    person_detector: Group[inference.detection.Settings]   = Group(inference.detection.Settings)   # the replay backend's person detector
    recorder       : Group[RecorderSettings]               = Group(RecorderSettings, share=[num_cams, fps, keyframe_interval])
    tracker        : Group[OnePerCamTrackerSettings]       = Group(OnePerCamTrackerSettings)
    frame_sync     : Group[SyncSettings]                   = Group(SyncSettings, share=[num_cams, fps])
    tracklet_sync  : Group[SyncSettings]                   = Group(SyncSettings, share=[num_cams, fps])

    @property
    def cameras(self) -> list[OakCameraSettings]:
        return [self.cam_0]


# ---------------------------------------------------------------------------
#  InOut group (OSC sound + OSC control)
# ---------------------------------------------------------------------------

class _OscSoundSettings(OscSoundSettings):
    stage: Field[Stage] = Field(Stage.LERP)


class InOutGroup(BaseSettings):
    osc_sound  : Group[_OscSoundSettings]   = Group(_OscSoundSettings)
    osc_control: Group[OscReceiverSettings] = Group(OscReceiverSettings)


# ---------------------------------------------------------------------------
#  Pose feature groups
# ---------------------------------------------------------------------------

class BboxFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class PointFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    confidence_filter: Group[nodes.DualConfidenceFilterSettings] = Group(nodes.DualConfidenceFilterSettings)
    smoother         : Group[nodes.EuroSmootherSettings]         = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction       : Group[nodes.PredictorSettings]            = Group(nodes.PredictorSettings, share=[frequency])
    interpolator     : Group[nodes.ChaseInterpolatorSettings]    = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class AngleFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings, share=[frequency])
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky      : Group[nodes.StickyFillerSettings]      = Group(nodes.StickyFillerSettings)

class VelocityFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    extractor   : Group[nodes.AngleVelocityExtractorSettings] = Group(nodes.AngleVelocityExtractorSettings, share=[frequency])
    smoother    : Group[nodes.EuroSmootherSettings]           = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction  : Group[nodes.PredictorSettings]              = Group(nodes.PredictorSettings, share=[frequency])
    interpolator: Group[nodes.ChaseInterpolatorSettings]      = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky      : Group[nodes.StickyFillerSettings]           = Group(nodes.StickyFillerSettings)

class MotionFeature(BaseSettings):
    extractor     : Group[nodes.AngleMotionExtractorSettings]  = Group(nodes.AngleMotionExtractorSettings)
    moving_average: Group[nodes.MovingAverageSmootherSettings] = Group(nodes.MovingAverageSmootherSettings)


# ---------------------------------------------------------------------------
#  Pose pipeline group (no similarity — single player)
# ---------------------------------------------------------------------------

class PoseGroup(BaseSettings):
    max_players:          Field[int]                  = Field(1, min=1, max=16, access=Field.INIT)
    model_type:         Field[inference.ModelType]  = Field(inference.ModelType.TRT, access=Field.INIT)
    model_path:         Field[str]                  = Field("", access=Field.INIT, visible=False)
    verbose:            Field[bool]                 = Field(False, access=Field.INIT)
    frequency:          Field[float]                = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float]                = Field(60.0)

    _batch_share  : list = [max_players, model_type, model_path, verbose]
    _feature_share: list = [frequency, output_frequency]

    pose           : Group[inference.pose.Settings]          = Group(inference.pose.Settings, share=_batch_share)
    segmentation   : Group[inference.segmentation.Settings]  = Group(inference.segmentation.Settings, share=_batch_share)
    optical_flow   : Group[inference.optical_flow.Settings]  = Group(inference.optical_flow.Settings, share=_batch_share)
    tracklets      : Group[PosesFromTrackletsSettings]       = Group(PosesFromTrackletsSettings)
    image_crop     : Group[inference.crop.Settings]          = Group(inference.crop.Settings, share=[max_players])
    angle_extractor: Group[nodes.AngleExtractorSettings]     = Group(nodes.AngleExtractorSettings)
    bbox           : Group[BboxFeature]                      = Group(BboxFeature, share=_feature_share)
    point          : Group[PointFeature]                     = Group(PointFeature, share=_feature_share)
    angle          : Group[AngleFeature]                     = Group(AngleFeature, share=_feature_share)
    velocity       : Group[VelocityFeature]                  = Group(VelocityFeature, share=_feature_share)
    motion         : Group[MotionFeature]                    = Group(MotionFeature)
    motion_gate    : Group[nodes.MotionGateApplicatorSettings] = Group(nodes.MotionGateApplicatorSettings, share=[max_players])
    window_raw     : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_clean   : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_smooth  : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_predict : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_lerp    : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)


# ---------------------------------------------------------------------------
#  Render settings
# ---------------------------------------------------------------------------

# Stage-aware subclasses — override Field[int] with Field[Stage] for dropdown UI
class _DataLayerSettings(layers.DataLayerSettings):
    stage: Field[Stage] = Field(Stage.SMOOTH)

class _TrackerCompositorSettings(layers.TrackerCompositorSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for pose data")

class _PoseCompositorSettings(layers.PoseCompositorSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for camera crop")

class _CenterGeometrySettings(layers.CenterGeometrySettings):
    stage: Field[Stage] = Field(Stage.SMOOTH, description="Pose data pipeline stage")

class FluidInputSettings(BaseSettings):
    flow:    Field[Layers] = Field(Layers.center_mask,       description="Texture used as optical flow input")
    density: Field[Layers] = Field(Layers.center_foreground, description="Texture used as density and temperature input")

class _FlowLayerSettings(layers.FlowLayerSettings):
    inputs: Group[FluidInputSettings] = Group(FluidInputSettings)


class LayerGroup(BaseSettings):
    select   : Group[LayerSettings]                 = Group(LayerSettings)
    composite: Group[layers.CompositeLayerSettings] = Group(layers.CompositeLayerSettings)

class DataGroup(BaseSettings):
    a: Group[_DataLayerSettings] = Group(_DataLayerSettings)
    b: Group[_DataLayerSettings] = Group(_DataLayerSettings)

class PreviewGroup(BaseSettings):
    tracker: Group[_TrackerCompositorSettings] = Group(_TrackerCompositorSettings)
    pose   : Group[_PoseCompositorSettings]    = Group(_PoseCompositorSettings)

class CenterGroup(BaseSettings):
    geometry  : Group[_CenterGeometrySettings]              = Group(_CenterGeometrySettings)
    mask      : Group[layers.CenterMaskLayerSettings]       = Group(layers.CenterMaskLayerSettings)
    cam       : Group[layers.CenterCamLayerSettings]        = Group(layers.CenterCamLayerSettings)
    foreground: Group[layers.CenterForegroundLayerSettings] = Group(layers.CenterForegroundLayerSettings)
    pose      : Group[layers.CenterPoseLayerSettings]       = Group(layers.CenterPoseLayerSettings)
    color     : Group[layers.MSColorMaskLayerSettings]      = Group(layers.MSColorMaskLayerSettings)

class RenderSettings(BaseSettings):
    stage  : Field[Stage]                        = Field(Stage.LERP, description="Pipeline stage for flow/fluid/color layers")
    layer  : Group[LayerGroup]                   = Group(LayerGroup)
    data   : Group[DataGroup]                    = Group(DataGroup)
    preview: Group[PreviewGroup]                 = Group(PreviewGroup)
    center : Group[CenterGroup]                  = Group(CenterGroup)
    flow   : Group[_FlowLayerSettings]           = Group(_FlowLayerSettings, share=[stage])
    fluid3d: Group[layers.Fluid3DLayerSettings]  = Group(layers.Fluid3DLayerSettings, share=[stage])
    colors : Group[ColorSettings]                = Group(ColorSettings)
    window : Group[WindowSettings]               = Group(WindowSettings)


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    max_players:        Field[int]   = Field(1, access=Field.INIT, visible=False)
    num_cams:           Field[int]   = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    input_fps:          Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)

    camera: Group[OakGroup]       = Group(OakGroup, share=[num_cams, input_fps.as_('fps')])
    pose  : Group[PoseGroup]      = Group(PoseGroup, share=[max_players, input_fps.as_('frequency')])
    render: Group[RenderSettings] = Group(RenderSettings)
    inout : Group[InOutGroup]     = Group(InOutGroup)
    panel : Group[PanelSettings]  = Group(PanelSettings)
