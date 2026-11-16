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
from modules.camera import CameraHealthSettings, CameraResolution
from modules.camera.oak4 import Oak4CameraSettings
from modules.recording import PlayerSettings, RecorderSettings
from modules.session import SessionSettings
from modules import render
from modules.inout import OscSoundSettings, OscReceiverSettings
from modules.tracker import OnePerCamTrackerSettings, TrackerFrameSourceSettings
from modules.pose import nodes, trackers, window
from modules import inference
from modules.gl import Colormap, WindowSettings
from modules.flow import DepthRead
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
    """What a panel can show, in the chain's order: each stage's monitor, the show, the timings and the pose."""
    capture =       0
    view =          auto()
    motion =        auto()
    bridge =        auto()
    fluid =         auto()
    show =          auto()     # the fluid's output, its look
    time_series =   auto()     # the app's timings, over time
    pose =          auto()

class MainView(IntEnum):
    """What the main window shows: the debug panels, or the show alone."""
    DEBUG = 0       # panel a and panel b, as `a` and `b` pick them
    SHOW = auto()   # panel a's layers alone, fit to the whole window, as a secondary window shows it


class LayerSettings(BaseSettings):
    """The two panels' layers, each drawn in order: `a` is the show and what the screen recording records."""
    a: Field[list[Layers]] = Field([Layers.show], description="Layers drawn in panel a, the show", widget=Widget.playlist)
    b: Field[list[Layers]] = Field([Layers.pose], description="Layers drawn in panel b", widget=Widget.playlist)


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
    camera_health  : Group[_CameraHealthSettings]          = Group(_CameraHealthSettings)
    simulator      : Group[PlayerSettings]                 = Group(PlayerSettings, share=[num_cams, fps])
    uploader       : Group[inference.SerialUploaderSettings] = Group(inference.SerialUploaderSettings)
    person_detector: Group[inference.detection.Settings]   = Group(inference.detection.Settings)   # the replay backend's person detector
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
    model_path:         Field[str]                  = Field("", access=Field.INIT, visible=False)
    verbose:            Field[bool]                 = Field(False, access=Field.INIT)
    frequency:          Field[float]                = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float]                = Field(60.0)

    _batch_share  : list = [max_players, model_type, model_path, verbose]
    _feature_share: list = [frequency, output_frequency]

    pose           : Group[inference.pose.Settings]          = Group(inference.pose.Settings, share=_batch_share)
    optical_flow   : Group[inference.optical_flow.Settings]  = Group(inference.optical_flow.Settings, share=_batch_share)
    frame_source   : Group[TrackerFrameSourceSettings]       = Group(TrackerFrameSourceSettings)
    pose_crop      : Group[inference.crop.Settings]          = Group(inference.crop.Settings, share=[max_players])
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
    """The view's crop: the part of the camera image the view shows, its center and its height."""
    center_x: Field[float] = Field(0.5, min=0.0, max=1.0, description="The crop's center across the camera image, a share of its width")
    center_y: Field[float] = Field(0.5, min=0.0, max=1.0, description="The crop's center down the camera image, a share of its height")
    height:   Field[float] = Field(1.0, min=0.05, max=2.0, description="The crop's height, a share of the camera image's")


class ViewDepthSettings(BaseSettings):
    """ViewDepth: how the depth is read into the view and the tolerance it reads at."""
    depth_read: Field[DepthRead] = Field(DepthRead.SURFACE, description="How the depth is read into the view: its edge on its own pixels or on the mask's")
    tolerance:  Field[float]     = Field(0.2, min=0.05, max=0.5, step=0.01, description="Depth jump beyond which the read blends no two depth pixels, m")


class ViewDepthMaskSettings(BaseSettings):
    """ViewDepthMask: the player's depth band, the cut; the guided filter that moves the cut's edge onto the image's."""
    near:          Field[float] = Field(0.5, min=0.1, max=12.0, step=0.05, description="Nearest depth that is the player (m)")
    far:           Field[float] = Field(4.0, min=0.1, max=12.0, step=0.05, description="Farthest depth that is the player (m)")
    guided_radius: Field[float] = Field(2.0, min=0.0, max=8.0, step=0.25, newline=True, description="How far the image moves the mask's edge to its own, in depth pixels; 0 = off")
    edge_contrast: Field[float] = Field(0.03, min=0.001, max=0.3, step=0.001, description="The image's brightness spread (0 to 1) above which the mask's edge follows the image's")


class ColorGradeSettings(BaseSettings):
    """ColorGrade: the image graded, in the order applied, then sharpened, the mask as its alpha."""
    enabled:    Field[bool]  = Field(True, description="The color graded; off, the image as it is")
    use_mask:   Field[bool]  = Field(True, description="The mask as the color's alpha")
    exposure:   Field[float] = Field(1.0, min=0.1, max=3.0, newline=True, description="The color times this")
    offset:     Field[float] = Field(0.0, min=-0.5, max=0.5, description="Added to the color, after the exposure")
    gamma:      Field[float] = Field(1.0, min=0.5, max=2.0, description="Gamma")
    contrast:   Field[float] = Field(1.0, min=0.5, max=2.0, description="Contrast around mid grey")
    saturation: Field[float] = Field(1.0, min=0.0, max=2.0, description="Saturation, times this")
    sharpen:    Field[float] = Field(0.0, min=0.0, max=2.0, description="Sharpen (0 = off)")


class ViewLayerSettings(render.StageLayerSettings):
    """The view's size; its crop, depth, depth mask and color grade in their panels, in the order the view runs them."""
    width:         Field[int]   = Field(448, min=32, max=4096, step=32, description="The view's width (pixels)")
    height:        Field[int]   = Field(800, min=32, max=4096, step=32, description="The view's height (pixels)")
    crop:        Group[ViewCropSettings]      = Group(ViewCropSettings)
    depth:       Group[ViewDepthSettings]     = Group(ViewDepthSettings)
    depth_mask:  Group[ViewDepthMaskSettings] = Group(ViewDepthMaskSettings)
    color_grade: Group[ColorGradeSettings]    = Group(ColorGradeSettings)


class CaptureView(IntEnum):
    """What the capture's monitor shows."""
    COLOR = 0       # the capture's color image
    DEPTH = 1       # its depth, in the colormap
    BOTH = 2        # its depth in the colormap over its color image


class CaptureSettings(BaseSettings):
    """The capture's monitor: the color, the depth or both, the depth in `colormap` from the view's near to its far."""
    show:     Field[CaptureView] = Field(CaptureView.BOTH, description="What is shown: the color, the depth or both")
    colormap: Field[Colormap]    = Field(Colormap.JET, description="The depth's colors, from the view's near to its far")


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
    main_view: Field[MainView]                   = Field(MainView.DEBUG, description="The main window: the debug panels, or the show alone")
    input_fps: Field[float]                      = Field(60.0, min=1.0, max=120.0, access=Field.INIT, visible=False, description="Camera frame rate")
    layer  : Group[LayerSettings]                = Group(LayerSettings)
    # the chain, in its order, then the pose beside it
    capture: Group[CaptureSettings]              = Group(CaptureSettings)
    view   : Group[ViewLayerSettings]            = Group(ViewLayerSettings)
    motion : Group[render.Motion3DLayerSettings] = Group(render.Motion3DLayerSettings, share=[input_fps.as_('fps')])
    bridge : Group[render.Bridge3DLayerSettings] = Group(render.Bridge3DLayerSettings)
    fluid  : Group[render.Fluid3DLayerSettings]  = Group(render.Fluid3DLayerSettings)
    pose   : Group[PoseMonitorSettings]          = Group(PoseMonitorSettings)
    # the display
    time_series: Group[render.TimeSeriesLayerSettings] = Group(render.TimeSeriesLayerSettings)
    colors : Group[render.ColorSettings]         = Group(render.ColorSettings)
    screen_recorder: Group[ScreenRecorderSettings] = Group(ScreenRecorderSettings)
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
    pose_drops:      Field[int]   = Field(0, access=Field.READ, pinned=True, label="Pose drops",
                                          description="Pose model results dropped before the stages after it took them")


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
