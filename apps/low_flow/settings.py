"""Low Flow settings — single-camera 2D fluid installation.

PRESET MAINTENANCE
------------------
Preset JSON files live in ``apps/low_flow/data/settings/``.
Each JSON mirrors this settings tree exactly.  When you rename, add,
or remove a Field here, update every ``.json`` file in that directory
to match — delete stale keys, add new keys with their Field default.
The root class is ``Settings``.
"""

from enum import IntEnum, auto

from modules.settings import BaseSettings, Field, Group, Widget
from modules.panel import PanelSettings
from modules.camera import CameraHealthSettings, CameraResolution
from modules.camera.oak4 import Oak4CameraSettings
from modules.recording import PlayerSettings, RecorderSettings
from modules.session import SessionSettings
from modules import render
from modules.inout import OscSoundSettings, OscReceiverSettings
from modules.tracker import OnePerCamTrackerSettings, TrackerFrameSourceSettings
from modules.pose import nodes, window
from modules import inference
from modules.gl import WindowManagerSettings, WindowSettings
from modules.video import ScreenRecorderSettings


# ---------------------------------------------------------------------------
#  Pipeline stages
# ---------------------------------------------------------------------------

class Stage(IntEnum):
    RAW =       0
    CLEAN =     auto()
    SMOOTH =    auto()
    LERP =      auto()


# ---------------------------------------------------------------------------
#  Layer selection settings (split view)
# ---------------------------------------------------------------------------

class Layers(IntEnum):
    """What a panel can show, in the chain's order: each stage's monitor, the timings and the pose."""
    inference_frame = 0
    view =          auto()
    color =         auto()
    motion =        auto()
    bridge =        auto()
    fluid =         auto()
    show =          auto()     # the fluid's output, its look
    time_series =   auto()     # the app's timings, over time
    pose =          auto()

class LayerSettings(BaseSettings):
    """The two panels' layers, each drawn in order: `main` is the show and what the screen recording records."""
    main:  Field[list[Layers]] = Field([Layers.inference_frame], description="Layers drawn in the main panel, the show", widget=Widget.playlist)
    debug: Field[list[Layers]] = Field([Layers.pose], description="Layers drawn in the debug panel", widget=Widget.playlist)


# ---------------------------------------------------------------------------
#  OAK4 camera group (1 camera)
# ---------------------------------------------------------------------------

class _CameraHealthSettings(CameraHealthSettings):
    """No mount verdict here: tilt and roll are read in the camera's own `readings`."""
    mount:           Field[bool]  = Field(True, access=Field.READ, visible=False)
    mount_tolerance: Field[float] = Field(2.0, visible=False)


class Oak4Group(BaseSettings):
    num_cams:       Field[int]               = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    fps:                Field[float]             = Field(60.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    keyframe_interval:  Field[int]               = Field(60, access=Field.INIT, visible=False, description="Keyframe interval in frames")
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
    camera_health  : Group[_CameraHealthSettings]          = Group(_CameraHealthSettings)
    simulator      : Group[PlayerSettings]                 = Group(PlayerSettings, share=[num_cams, fps])
    uploader       : Group[inference.SerialUploaderSettings] = Group(inference.SerialUploaderSettings)
    person_detector: Group[inference.person.PersonDetectorSettings]   = Group(inference.person.PersonDetectorSettings)   # the replay backend's person detector
    recorded_poses : Group[inference.RecordedPosesSettings] = Group(inference.RecordedPosesSettings)   # the replay's poses in place of the pose model
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
    onnx_path:          Field[str]                  = Field("data/models", access=Field.INIT, visible=False)
    engine_path:        Field[str]                  = Field("data/models", access=Field.INIT, visible=False)
    verbose:            Field[bool]                 = Field(False, access=Field.INIT)
    frequency:          Field[float]                = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float]                = Field(60.0)

    _batch_share  : list = [max_players, model_type, onnx_path, engine_path, verbose]
    _feature_share: list = [frequency, output_frequency]

    pose           : Group[inference.pose.PoseEstimatorSettings]          = Group(inference.pose.PoseEstimatorSettings, share=_batch_share)
    segmentation   : Group[inference.segmentation.SegmenterSettings]      = Group(inference.segmentation.SegmenterSettings, share=_batch_share)
    frame_source   : Group[TrackerFrameSourceSettings]       = Group(TrackerFrameSourceSettings)
    pose_crop      : Group[inference.CropExtractorSettings]          = Group(inference.CropExtractorSettings, share=[max_players])
    segmentation_crop: Group[inference.CropExtractorSettings]        = Group(inference.CropExtractorSettings, share=[max_players])
    inference_frame_sync   : Group[inference.InferenceFrameSyncSettings]    = Group(inference.InferenceFrameSyncSettings)
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
class _DataLayerSettings(render.DataLayerSettings):
    stage: Field[Stage] = Field(Stage.SMOOTH)

class _TrackerCompositorSettings(render.TrackerCompositorSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for pose data")
    box_stage: Field[Stage] = Field(Stage.RAW, description="Pipeline stage of the bounding boxes")
    image: Field[bool] = Field(False, description="The camera image under them; off, they lie over any layer of its frame")

class _PoseCompositorSettings(render.PoseCompositorSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for camera crop")
    back_stage: Field[Stage] = Field(Stage.CLEAN, description="Pipeline stage of the back line (black, 3x wide)")
    middle_stage: Field[Stage] = Field(Stage.SMOOTH, description="Pipeline stage of the middle line (white, 2x wide)")
    front_stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage of the front line (track color)")


# The view's layer's settings (render/layers/view)
class ViewCropSettings(BaseSettings):
    """ViewCrop, the window the view cuts from the camera image: its center and its height."""
    center_x: Field[float] = Field(0.5, min=0.0, max=1.0, description="The crop's center across the camera image, a share of its width")
    center_y: Field[float] = Field(0.5, min=0.0, max=1.0, description="The crop's center down the camera image, a share of its height")
    height:   Field[float] = Field(1.0, min=0.05, max=2.0, description="The crop's height, a share of the camera image's")


class ViewLayerSettings(render.StageLayerSettings):
    """The view's size and its crop."""
    width:         Field[int]   = Field(448, min=32, max=4096, step=32, description="The view's width (pixels)")
    height:        Field[int]   = Field(800, min=32, max=4096, step=32, description="The view's height (pixels)")
    crop:        Group[ViewCropSettings]      = Group(ViewCropSettings)

    monitor_main:  Group[render.Monitor2DLayerSettings] = Group(render.Monitor2DLayerSettings, last=True, tint=render.MAIN_TINT)
    monitor_debug: Group[render.Monitor2DLayerSettings] = Group(render.Monitor2DLayerSettings, last=True, tint=render.DEBUG_TINT)


class PoseFrame(IntEnum):
    """Where the pose monitor draws the pose."""
    CROP = 0        # on the crop, with the crop's image: the pose model's input
    CAMERA = 1      # on the camera's frame, with its boxes; no image
    VIEW = 2        # on the view; no image


class PoseFeatures(IntEnum):
    """The features the pose monitor draws over the pose."""
    NONE = 0
    FRAME = 1       # the current frame's
    WINDOW = 2      # the window's, over time


class PoseMonitorSettings(BaseSettings):
    """The pose monitor: the pose stage it reads, where it draws the pose and the features over it; how each is
    drawn in its group."""
    stage:    Field[Stage]        = Field(Stage.LERP, description="The pose stage drawn")
    frame:    Field[PoseFrame]    = Field(PoseFrame.CAMERA, description="Where the pose is drawn")
    features: Field[PoseFeatures] = Field(PoseFeatures.NONE, description="The features drawn over the pose")

    crop    : Group[_PoseCompositorSettings]    = Group(_PoseCompositorSettings, share=[stage])
    camera  : Group[_TrackerCompositorSettings] = Group(_TrackerCompositorSettings, share=[stage])
    view    : Group[render.ViewPoseLayerSettings] = Group(render.ViewPoseLayerSettings)
    data    : Group[_DataLayerSettings]         = Group(_DataLayerSettings, share=[stage])

class RenderSettings(BaseSettings):
    """The render's panels and its monitors beside the flow: the pose and the timings."""
    debug: Field[bool]                           = Field(True, description="The main and the debug panel; off, the main panel alone, as on a second screen")
    freeze_frame: Field[bool]                    = Field(False, description="Hold the last frame; the view and the color are made again on it when a setting changes")
    layer  : Group[LayerSettings]                = Group(LayerSettings)
    pose   : Group[PoseMonitorSettings]          = Group(PoseMonitorSettings)
    # the display
    time_series: Group[render.TimeSeriesLayerSettings] = Group(render.TimeSeriesLayerSettings)
    colors : Group[render.ColorSettings]         = Group(render.ColorSettings)


class CadenceSettings(BaseSettings):
    """Pinned beside the camera's video rates: the frames that get past the camera, so a frame lost on its
    way to the flow shows where (the sync's frames, then the motion's measures), with the drops counted where they happen."""
    inference_frame_fps:     Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Inference frames", unit="fps",
                                          description="Inference frames the sync hands on, average")
    inference_frame_min_fps: Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Inference frames min", unit="fps",
                                          description="Inference frames: 1 / the worst gap between two")
    motion_fps:      Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Motion", unit="fps",
                                          description="Frames the motion measures, average")
    motion_min_fps:  Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Motion min", unit="fps",
                                          description="Motion: 1 / the worst gap between two measures")
    upload_drops:    Field[int]   = Field(0, access=Field.READ, pinned=True, label="Upload drops",
                                          description="Camera frames the uploader dropped, its queue full")
    inference_frame_drops:   Field[int]   = Field(0, access=Field.READ, pinned=True, label="Inference frame drops",
                                          description="Inference frames dropped before the render took them")
    pose_drops:      Field[int]   = Field(0, access=Field.READ, pinned=True, label="Pose drops",
                                          description="Pose model results dropped before the stages after it took them")


class WindowsSettings(WindowManagerSettings):
    """The render's window"""
    window_0: Group[WindowSettings]        = Group(WindowSettings)


# ---------------------------------------------------------------------------
#  Root groups
# ---------------------------------------------------------------------------

class SetupSettings(BaseSettings):
    """The machine and the site: the camera, the pose, the render, the window and the links."""
    max_players:        Field[int]   = Field(1, access=Field.INIT, visible=False)
    num_cams:           Field[int]   = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    input_fps:          Field[float] = Field(60.0, min=1.0, max=120.0, access=Field.INIT, visible=False, description="Camera frame rate")

    camera: Group[Oak4Group]      = Group(Oak4Group, share=[num_cams, input_fps.as_('fps')])
    cadence: Group[CadenceSettings] = Group(CadenceSettings)
    pose  : Group[PoseGroup]      = Group(PoseGroup, share=[max_players, input_fps.as_('frequency')])
    render: Group[RenderSettings] = Group(RenderSettings)
    windows: Group[WindowsSettings] = Group(WindowsSettings)
    screen_recorder: Group[ScreenRecorderSettings] = Group(ScreenRecorderSettings)
    inout : Group[InOutGroup]     = Group(InOutGroup)
    panel : Group[PanelSettings]  = Group(PanelSettings)


class FlowSettings(BaseSettings):
    """The flow chain, in its order: the view, the color, the motion, the bridge and the fluid."""
    input_fps: Field[float]                      = Field(60.0, min=1.0, max=120.0, access=Field.INIT, visible=False, description="Camera frame rate")
    view   : Group[ViewLayerSettings]            = Group(ViewLayerSettings)
    color  : Group[render.ColorLayerSettings]    = Group(render.ColorLayerSettings)
    motion : Group[render.Motion2DLayerSettings] = Group(render.Motion2DLayerSettings, share=[input_fps.as_('fps')])
    bridge : Group[render.Bridge2DLayerSettings] = Group(render.Bridge2DLayerSettings)
    fluid  : Group[render.Fluid2DLayerSettings]  = Group(render.Fluid2DLayerSettings)


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    input_fps: Field[float] = Field(60.0, min=1.0, max=120.0, access=Field.INIT)

    setup: Group[SetupSettings] = Group(SetupSettings, share=[input_fps])
    flow : Group[FlowSettings]  = Group(FlowSettings, share=[input_fps])
