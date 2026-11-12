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
from modules.camera import CameraCheckSettings, CameraResolution, PlayerSettings, RecorderSettings
from modules.camera.oak4 import Oak4CameraSettings
from modules.session import SessionSettings
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
    LERP =      auto()


# ---------------------------------------------------------------------------
#  Layers enum
# ---------------------------------------------------------------------------

class Layers(IntEnum):
    # source layers
    cam_image =     0
    cam_crop =      auto()
    capture_image = auto()
    capture_depth = auto()
    # composite layers
    tracker =       auto()
    pose =          auto()
    # the view stage and its outputs
    view =          auto()
    view_image =    auto()
    view_foreground = auto()
    view_depth =    auto()
    view_mask =     auto()
    view_depth_mask = auto()
    view_pose =     auto()
    foreground_look = auto()
    # data layers
    time_series =   auto()
    data_B_W =      auto()
    data_B_F =      auto()
    data_A_W =      auto()
    data_A_F =      auto()
    data_time =     auto()
    # composition layers
    color_mask =    auto()
    motion =        auto()
    bridge =        auto()
    fluid =         auto()
    monitor_a =     auto()
    monitor_b =     auto()
    composite =     auto()


# ---------------------------------------------------------------------------
#  Layer selection settings (split view)
# ---------------------------------------------------------------------------

class MainView(IntEnum):
    """What the main window shows: the debug panels, or the show alone."""
    DEBUG = 0       # the left and the right panel, as `left` and `right` pick them
    SHOW = auto()   # the composite alone, fit to the whole window, as a secondary window shows it


class LayerSettings(BaseSettings):
    left:   Field[list[Layers]] = Field([Layers.composite],                  description="Layers drawn in the left viewport",  widget=Widget.playlist)
    right:  Field[list[Layers]] = Field([Layers.pose],                       description="Layers drawn in the right viewport", widget=Widget.playlist)
    output: Field[list[Layers]] = Field([Layers.fluid, Layers.color_mask], description="Layers mixed into the show texture", widget=Widget.playlist)


# ---------------------------------------------------------------------------
#  OAK4 camera group (1 camera)
# ---------------------------------------------------------------------------

class _CameraCheckSettings(CameraCheckSettings):
    """No mount verdict here: tilt and roll are read in the camera's own `readings`."""
    mount:           Field[bool]  = Field(True, access=Field.READ, visible=False)
    mount_tolerance: Field[float] = Field(2.0, visible=False)


class Oak4Group(BaseSettings):
    num_cams:       Field[int]               = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    fps:                Field[float]             = Field(60.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    keyframe_interval:  Field[int]               = Field(60, access=Field.INIT, visible=False, description="Keyframe interval in frames, the same for colour and depth")
    color:              Field[bool]              = Field(True, access=Field.INIT, description="Enable color capture")
    resolution:         Field[CameraResolution]  = Field(CameraResolution.P1080, access=Field.INIT, description="Delivered size on USB; the sensor mode follows it")
    sim_enabled:        Field[bool]              = Field(False, access=Field.INIT, description="Enable simulation mode")
    model_path:         Field[str]               = Field("data/models", access=Field.INIT, visible=False, description="Model files directory")
    fov:                Field[float]             = Field(80.0, access=Field.INIT, description="Field (°) the grey frame's warp spans across the sensor width; color brings its own lens")

    start:              Field[bool]              = Field(False, widget=Widget.button, description="Start recording")
    stop:               Field[bool]              = Field(False, widget=Widget.button, description="Stop recording")
    split:              Field[bool]              = Field(False, widget=Widget.button, description="Split chunk", visible=False)
    split_seconds:      Field[float]             = Field(60, min=1, max=60, widget=Widget.number, description="Split recording into chunks of this length (seconds)")

    _cam_share: list = [fps, color, resolution, model_path, fov]
    cam_0          : Group[Oak4CameraSettings]             = Group(Oak4CameraSettings, share=_cam_share)
    camera_check   : Group[_CameraCheckSettings]           = Group(_CameraCheckSettings)
    simulator      : Group[PlayerSettings]                 = Group(PlayerSettings, share=[num_cams, fps])
    person_detector: Group[inference.detection.Settings]   = Group(inference.detection.Settings)   # the replay backend's person detector
    control        : Group[SessionSettings]                = Group(SessionSettings, share=[start, stop, split, split_seconds])
    recorder       : Group[RecorderSettings]               = Group(RecorderSettings, share=[num_cams, fps, keyframe_interval, start, stop, split])
    tracker        : Group[OnePerCamTrackerSettings]       = Group(OnePerCamTrackerSettings)

    @property
    def cameras(self) -> list[Oak4CameraSettings]:
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

    smoother    : Group[nodes.EuroSmootherSettings]        = Group(nodes.EuroSmootherSettings)
    prediction  : Group[nodes.PredictorSettings]           = Group(nodes.PredictorSettings)
    interpolator: Group[nodes.PredictInterpolatorSettings] = Group(nodes.PredictInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class PointFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    confidence_filter: Group[nodes.DualConfidenceFilterSettings] = Group(nodes.DualConfidenceFilterSettings)
    smoother         : Group[nodes.EuroSmootherSettings]         = Group(nodes.EuroSmootherSettings)
    interpolator     : Group[nodes.PredictInterpolatorSettings]  = Group(nodes.PredictInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class AngleFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    smoother    : Group[nodes.EuroSmootherSettings]        = Group(nodes.EuroSmootherSettings)
    interpolator: Group[nodes.PredictInterpolatorSettings] = Group(nodes.PredictInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky      : Group[nodes.StickyFillerSettings]        = Group(nodes.StickyFillerSettings)

class VelocityFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    extractor   : Group[nodes.AngleVelocityExtractorSettings] = Group(nodes.AngleVelocityExtractorSettings)
    smoother    : Group[nodes.EuroSmootherSettings]           = Group(nodes.EuroSmootherSettings)
    interpolator: Group[nodes.PredictInterpolatorSettings]    = Group(nodes.PredictInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
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
    optical_flow   : Group[inference.optical_flow.Settings]  = Group(inference.optical_flow.Settings, share=_batch_share)
    tracklets      : Group[PosesFromTrackletsSettings]       = Group(PosesFromTrackletsSettings)
    image_crop     : Group[inference.crop.Settings]          = Group(inference.crop.Settings, share=[max_players])
    capture_sync   : Group[inference.CaptureSyncSettings]    = Group(inference.CaptureSyncSettings)
    angle_extractor: Group[nodes.AngleExtractorSettings]     = Group(nodes.AngleExtractorSettings)
    bbox           : Group[BboxFeature]                      = Group(BboxFeature, share=_feature_share)
    point          : Group[PointFeature]                     = Group(PointFeature, share=_feature_share)
    angle          : Group[AngleFeature]                     = Group(AngleFeature, share=_feature_share)
    velocity       : Group[VelocityFeature]                  = Group(VelocityFeature, share=_feature_share)
    motion         : Group[MotionFeature]                    = Group(MotionFeature)
    window_raw     : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_clean   : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_smooth  : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
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

class FlowInputSettings(BaseSettings):
    """Which layers feed the flow, chosen live, and how the fluid weights its injections."""
    motion_image:    Field[Layers] = Field(Layers.view_image,      description="The picture the motion is measured on")
    bridge_image:    Field[Layers] = Field(Layers.foreground_look, description="The picture the density's color comes from")
    scale_by_motion: Field[bool]   = Field(True, description="Weight each injection by the player's motion; off is full weight")


class LayerGroup(BaseSettings):
    select   : Group[LayerSettings]                 = Group(LayerSettings)
    composite: Group[layers.CompositeLayerSettings] = Group(layers.CompositeLayerSettings)

class DataGroup(BaseSettings):
    a: Group[_DataLayerSettings] = Group(_DataLayerSettings)
    b: Group[_DataLayerSettings] = Group(_DataLayerSettings)

class PreviewGroup(BaseSettings):
    tracker: Group[_TrackerCompositorSettings] = Group(_TrackerCompositorSettings)
    pose   : Group[_PoseCompositorSettings]    = Group(_PoseCompositorSettings)

class RenderSettings(BaseSettings):
    main_view: Field[MainView]                   = Field(MainView.DEBUG, description="The main window: the debug panels, or the show alone")
    stage  : Field[Stage]                        = Field(Stage.LERP, description="Pipeline stage for flow/fluid/color layers")
    input_fps: Field[float]                      = Field(60.0, min=1.0, max=120.0, access=Field.INIT, visible=False, description="Camera frame rate")
    layer  : Group[LayerGroup]                   = Group(LayerGroup)
    data   : Group[DataGroup]                    = Group(DataGroup)
    preview: Group[PreviewGroup]                 = Group(PreviewGroup)
    view_depth: Group[layers.ViewDepthLayerSettings] = Group(layers.ViewDepthLayerSettings)
    view_pose: Group[layers.CenterPoseLayerSettings] = Group(layers.CenterPoseLayerSettings)
    foreground_look: Group[layers.ForegroundLookSettings] = Group(layers.ForegroundLookSettings)
    color_mask: Group[layers.MSColorMaskLayerSettings] = Group(layers.MSColorMaskLayerSettings)
    inputs : Group[FlowInputSettings]            = Group(FlowInputSettings)
    view   : Group[layers.ViewLayerSettings]     = Group(layers.ViewLayerSettings)
    motion : Group[layers.Motion3DLayerSettings] = Group(layers.Motion3DLayerSettings, share=[input_fps.as_('fps')])
    bridge : Group[layers.Bridge3DLayerSettings] = Group(layers.Bridge3DLayerSettings)
    fluid  : Group[layers.Fluid3DLayerSettings]  = Group(layers.Fluid3DLayerSettings)
    monitor_a: Group[layers.MonitorLayerSettings] = Group(layers.MonitorLayerSettings)
    monitor_b: Group[layers.MonitorLayerSettings] = Group(layers.MonitorLayerSettings)
    time_series: Group[layers.TimeSeriesLayerSettings] = Group(layers.TimeSeriesLayerSettings)
    colors : Group[ColorSettings]                = Group(ColorSettings)
    window : Group[WindowSettings]               = Group(WindowSettings)


class CadenceSettings(BaseSettings):
    """Pinned beside the camera's video rates: the frames that get past the camera, so a frame lost on its
    way to the flow shows where (the sync's captures, then the motion's measures), with the drops counted where they happen."""
    capture_fps:     Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Captures", unit="fps",
                                          description="Captures the sync hands on, average")
    capture_min_fps: Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Captures min", unit="fps",
                                          description="Captures: 1 / the worst gap between two")
    motion_fps:      Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Motion", unit="fps",
                                          description="Frames the motion measures, average")
    motion_min_fps:  Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Motion min", unit="fps",
                                          description="Motion: 1 / the worst gap between two measures")
    upload_drops:    Field[int]   = Field(0, access=Field.READ, pinned=True, label="Upload drops",
                                          description="Camera frames the uploader dropped, its queue full")
    capture_drops:   Field[int]   = Field(0, access=Field.READ, pinned=True, label="Capture drops",
                                          description="Captures dropped before the render took them")


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    max_players:        Field[int]   = Field(1, access=Field.INIT, visible=False)
    num_cams:           Field[int]   = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    input_fps:          Field[float] = Field(60.0, min=1.0, max=120.0, access=Field.INIT)

    camera: Group[Oak4Group]      = Group(Oak4Group, share=[num_cams, input_fps.as_('fps')])
    cadence: Group[CadenceSettings] = Group(CadenceSettings)
    pose  : Group[PoseGroup]      = Group(PoseGroup, share=[max_players, input_fps.as_('frequency')])
    render: Group[RenderSettings] = Group(RenderSettings, share=[input_fps])
    inout : Group[InOutGroup]     = Group(InOutGroup)
    panel : Group[PanelSettings]  = Group(PanelSettings)
