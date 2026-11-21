"""HD Trio's settings tree, its root ``Settings``, which holds ``setup``; its files in ``apps/hd_trio/data/settings/setup/``."""

from enum import IntEnum, auto
from typing import Any

from modules.settings import BaseSettings, Field, Group, Widget
from modules.panel import PanelSettings
from modules.camera import CameraHealthSettings, CameraResolution
from modules.camera.oak import OakCameraSettings
from modules.recording import PlayerSettings, RecorderSettings
from modules import render
from modules.inout import OscSoundSettings, ArtNetBarsSettings, OscReceiverSettings
from modules.utils import Color
from modules.tracker import OnePerCamTrackerSettings, TrackerFrameSourceSettings
from modules.pose import nodes, trackers, window, analytics, RecorderSettings as PoseRecorderSettings
from modules import inference
from modules.session import SessionSettings
from modules.gl import BlendState
from modules.window import WindowManagerSettings, WindowSettings
from .flow import GradientFlowSettings, Fluid2DSettings, bridge_2d
from .sequencer import SequencerSettings


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
    """Show stages, in play order.

    The integer value is also the audio cue: it is sent verbatim on
    ``/global/state`` and the Max patch plays ``state{value}.wav``.
    Renumbering re-points every narration cue.
    """
    WELCOME_IN =    0
    WELCOME =       auto()
    MOVEMENT =      auto()
    WHITE_POSE =    auto()
    FLUID =         auto()
    PRACTICE =      auto()
    ENJOY_IN =      auto()
    ENJOY =         auto()
    PLAY =          auto()
    CONCLUSION =    auto()
    BLACKOUT =      auto()
    IDLE =          auto()

# IDLE is the resting state: the sequencer parks on the LAST playlist entry when
# the show stops, so it must stay last.
_SHOW_STAGES = list(ShowStage)
_STAGE_DURATIONS: dict[ShowStage, float] = {
    ShowStage.WELCOME_IN: 3.0,
    ShowStage.WELCOME: 5.0,
    ShowStage.MOVEMENT: 24.0,
    ShowStage.WHITE_POSE: 3.0,
    ShowStage.FLUID: 3.0,
    ShowStage.PRACTICE: 10.0,
    ShowStage.ENJOY_IN: 3.0,
    ShowStage.ENJOY: 10.0,
    ShowStage.PLAY: 170.0,
    ShowStage.CONCLUSION: 6.0,
    ShowStage.BLACKOUT: 12.0,
    ShowStage.IDLE: 6.0,
}


class ShowSequencerSettings(SequencerSettings):
    """HD Trio show sequencer with project-specific stages."""
    stages:     Field[list[ShowStage]] = Field(_SHOW_STAGES, widget=Widget.checklist, description="Stages to play")
    # Indexed by ShowStage value, not by playlist position — must cover every member.
    durations:  Field[list] = Field([_STAGE_DURATIONS[s] for s in ShowStage], min=0.0, max=600.0, step=0.1, description="Stage durations")
    stage:      Field[ShowStage]   = Field(ShowStage.IDLE, access=Field.READ, description="Current stage", newline=True)


# ---------------------------------------------------------------------------
#  Layers enum  (defines which render layers this app uses)
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
    cam_readings =  auto()   # each camera's frame rate, tilt and roll error, over its own view
    # center layers
    center_geometry = auto()
    center_cam =    auto()
    center_mask =   auto()
    center_foreground = auto()
    center_pose =   auto()
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
    motion =        auto()
    bridge =        auto()
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
    num_cams       : Field[int]             = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    fps               : Field[float]           = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    yolo              : Field[bool]            = Field(True, access=Field.INIT, description="Enable YOLO person detection")
    color             : Field[bool]            = Field(True, access=Field.INIT, description="Color camera (False = mono)")
    square            : Field[bool]            = Field(True, access=Field.INIT, description="Use square aspect ratio")
    resolution        : Field[CameraResolution] = Field(CameraResolution.P1080, access=Field.INIT, description="Sensor mode. P720 and P800 on both sensors, P1080 color only.")
    sim_enabled: Field[bool]            = Field(False, access=Field.INIT, description="Enable simulation mode")
    model_path        : Field[str]             = Field("data/models", access=Field.INIT, visible=False, description="Model files directory")
    fov               : Field[float]           = Field(95.0, access=Field.INIT, description="Camera horizontal FOV over the full frame width (°)")  # OAK-1 W, IMX378 color, 16:9; the square crop keeps the degrees per pixel

    _cam_share: list = [fps, color, square, yolo, resolution, model_path, fov]

    cam_0        : Group[OakCameraSettings]            = Group(OakCameraSettings, share=_cam_share)
    cam_1        : Group[OakCameraSettings]            = Group(OakCameraSettings, share=_cam_share)
    cam_2        : Group[OakCameraSettings]            = Group(OakCameraSettings, share=_cam_share)
    camera_health: Group[CameraHealthSettings]      = Group(CameraHealthSettings)
    simulator    : Group[PlayerSettings]         = Group(PlayerSettings, share=[num_cams, fps])
    person_detector     : Group[inference.person.PersonDetectorSettings] = Group(inference.person.PersonDetectorSettings)     # the replay backend's person detector
    tracker      : Group[OnePerCamTrackerSettings]  = Group(OnePerCamTrackerSettings)
    frame_sync   : Group[inference.CameraSyncSettings]      = Group(inference.CameraSyncSettings, share=[fps])
    tracklet_sync: Group[inference.CameraSyncSettings]      = Group(inference.CameraSyncSettings, share=[fps])

    @property
    def cameras(self) -> list[OakCameraSettings]:
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

    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings)
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings)
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class PointFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(60.0)

    confidence_filter: Group[nodes.DualConfidenceFilterSettings] = Group(nodes.DualConfidenceFilterSettings)
    smoother         : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings)
    prediction       : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings)
    interpolator     : Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class AngleFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(60.0)

    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings)
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings)
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky      : Group[nodes.StickyFillerSettings]      = Group(nodes.StickyFillerSettings)

class VelocityFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(60.0)

    extractor   : Group[nodes.AngleVelocityExtractorSettings] = Group(nodes.AngleVelocityExtractorSettings)
    smoother    : Group[nodes.EuroSmootherSettings]      = Group(nodes.EuroSmootherSettings)
    prediction  : Group[nodes.PredictorSettings]         = Group(nodes.PredictorSettings)
    interpolator: Group[nodes.ChaseInterpolatorSettings] = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky      : Group[nodes.StickyFillerSettings]      = Group(nodes.StickyFillerSettings)

class MotionFeature(BaseSettings):
    extractor     : Group[nodes.AngleMotionExtractorSettings] = Group(nodes.AngleMotionExtractorSettings)
    moving_average: Group[nodes.MovingAverageSmootherSettings] = Group(nodes.MovingAverageSmootherSettings)

class SimilarityFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(60.0)
    max_players       : Field[int]   = Field(3, min=1, max=16, access=Field.INIT)

    window_similarity    : Group[analytics.WindowSimilaritySettings]      = Group(analytics.WindowSimilaritySettings, share=[max_players])
    window_correlation   : Group[analytics.WindowCorrelationSettings]     = Group(analytics.WindowCorrelationSettings, share=[max_players])
    smoother             : Group[nodes.EuroSmootherSettings]          = Group(nodes.EuroSmootherSettings)
    interpolator         : Group[nodes.ChaseInterpolatorSettings]     = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky               : Group[nodes.SimilarityStickyFillerSettings] = Group(nodes.SimilarityStickyFillerSettings)


# ---------------------------------------------------------------------------
#  Pose pipeline group
# ---------------------------------------------------------------------------

class PoseGroup(BaseSettings):
    max_players       : Field[int]       = Field(3, min=1, max=16, access=Field.INIT)
    model_type      : Field[inference.ModelType] = Field(inference.ModelType.TRT, access=Field.INIT)
    onnx_path       : Field[str]       = Field("data/models", access=Field.INIT, visible=False)
    engine_path     : Field[str]       = Field("data/models", access=Field.INIT, visible=False)
    verbose         : Field[bool]      = Field(False, access=Field.INIT)
    frequency       : Field[float]     = Field(30.0, access=Field.INIT)
    output_frequency: Field[float]     = Field(60.0)

    _batch_share  : list = [max_players, model_type, onnx_path, engine_path, verbose]
    _feature_share: list = [frequency, output_frequency]

    pose            : Group[inference.pose.PoseEstimatorSettings]          = Group(inference.pose.PoseEstimatorSettings, share=_batch_share)
    segmentation    : Group[inference.segmentation.SegmenterSettings]  = Group(inference.segmentation.SegmenterSettings, share=_batch_share)
    frame_source    : Group[TrackerFrameSourceSettings]       = Group(TrackerFrameSourceSettings)
    pose_crop       : Group[inference.CropExtractorSettings]          = Group(inference.CropExtractorSettings, share=[max_players])
    segmentation_crop: Group[inference.CropExtractorSettings]         = Group(inference.CropExtractorSettings, share=[max_players])
    angle_extractor : Group[nodes.AngleExtractorSettings]     = Group(nodes.AngleExtractorSettings)
    bbox            : Group[BboxFeature]                      = Group(BboxFeature, share=_feature_share)
    point           : Group[PointFeature]                     = Group(PointFeature, share=_feature_share)
    angle           : Group[AngleFeature]                     = Group(AngleFeature, share=_feature_share)
    velocity        : Group[VelocityFeature]                  = Group(VelocityFeature, share=_feature_share)
    motion          : Group[MotionFeature]                    = Group(MotionFeature)
    similarity      : Group[SimilarityFeature]                = Group(SimilarityFeature, share=[frequency, output_frequency, max_players])
    window_raw      : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_clean    : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_smooth   : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_predict  : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)
    window_lerp     : Group[window.WindowNodeSettings]        = Group(window.WindowNodeSettings)


class _PoseRecorderSettings(PoseRecorderSettings):
    stage: Field[Stage] = Field(Stage.RAW)


class _IntroRecorderSettings(_PoseRecorderSettings):
    """Dedicated recorder for authoring the white example figure.

    Kept unshared from SessionGroup on purpose — its start/stop must not
    propagate to the sequencer and the session recorders.
    """
    stage:       Field[Stage] = Field(Stage.SMOOTH, description="Pipeline stage to record")
    output_path: Field[str]   = Field("intro", access=Field.INIT, description="Intro recordings directory")
    name:        Field[str]   = Field("intro", widget=Widget.input, description="Recording name suffix")


# ---------------------------------------------------------------------------
#  Session group (recording lifecycle: OSC, timer, video & pose recorders)
# ---------------------------------------------------------------------------

class SessionGroup(BaseSettings):
    """App session group — composes SessionSettings with app-specific recorders."""
    num_cams:   Field[int]   = Field(1, access=Field.INIT, description="Number of cameras")
    fps:           Field[float] = Field(30.0, access=Field.INIT, description="Camera frame rate")

    start:         Field[bool]  = Field(False, widget=Widget.button, description="Start session", pinned=True)
    stop:          Field[bool]  = Field(False, widget=Widget.button, description="Stop session")
    output_path:   Field[str]   = Field("recordings", access=Field.INIT, description="Recordings output directory")
    name:          Field[str]   = Field("", widget=Widget.input, description="Recording group ID", access=Field.READWRITE)
    split:         Field[bool]  = Field(False, widget=Widget.button, description="Split chunk", visible=False)
    split_seconds: Field[float] = Field(10, min=1, max=60, widget=Widget.number, description="Split recording into chunks of this length (seconds)")

    _session_share:  list = [output_path, name, start, stop, split, split_seconds]
    _recorder_share: list = [start, stop, split, name, output_path]

    osc:        Group[OscReceiverSettings]     = Group(OscReceiverSettings)
    control:    Group[SessionSettings]         = Group(SessionSettings, share=_session_share)
    sequencer:  Group[ShowSequencerSettings]   = Group(ShowSequencerSettings, share=[start, stop])
    video:      Group[RecorderSettings]        = Group(RecorderSettings, share=_recorder_share + [num_cams, fps])
    pose:       Group[_PoseRecorderSettings]   = Group(_PoseRecorderSettings, share=_recorder_share)


# ---------------------------------------------------------------------------
#  Render settings (layer settings, center, flow, fluid, colors, window)
# ---------------------------------------------------------------------------

# Stage-aware subclasses — override Field[int] with Field[Stage] for dropdown UI
class _DataLayerSettings(render.DataLayerSettings):
    stage: Field[Stage] = Field(Stage.SMOOTH)

class _MotionTimeDrawSettings(render.MotionTimeDrawSettings):
    stage: Field[Stage] = Field(Stage.RAW, description="Pipeline stage for pose data")

class _TrackerCompositorSettings(render.TrackerCompositorSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for pose data")
    box_stage: Field[Stage] = Field(Stage.RAW, description="Pipeline stage of the bounding boxes")

class _PoseCompositorSettings(render.PoseCompositorSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for camera crop")
    back_stage: Field[Stage] = Field(Stage.CLEAN, description="Pipeline stage of the back line (black, 3x wide)")
    middle_stage: Field[Stage] = Field(Stage.SMOOTH, description="Pipeline stage of the middle line (white, 2x wide)")
    front_stage: Field[Stage] = Field(Stage.PREDICT, description="Pipeline stage of the front line (track color)")

class SnapPoint(IntEnum):
    """Body landmark that stays pinned in the output frame."""
    SHOULDERS = 0
    HIPS      = 1
    FEET      = 2


class CenterGeometrySettings(BaseSettings):
    """Configuration for CenterGeometry anchor point positioning."""
    stage:              Field[Stage] = Field(Stage.SMOOTH, description="Pose data pipeline stage", access=Field.READWRITE)
    target_x:           Field[float] = Field(0.5, min=0.0, max=1.0, description="Horizontal position of the pinned point in the output")
    target_y:           Field[float] = Field(0.33, min=0.0, max=1.0, description="Vertical position of the pinned point in the output")
    target_size:        Field[float] = Field(0.27, min=0.01, max=1.0, description="Torso height as a fraction of output height (zoom)")
    dst_aspectratio:    Field[float] = Field(0.5625, min=0.5, max=2.0, description="Output aspect ratio (9/16 = 0.5625)")
    torso_upright:      Field[bool] = Field(True, description="Straighten the torso to vertical; off, it tilts with the body", newline=True)
    snap_point:         Field[SnapPoint] = Field(SnapPoint.SHOULDERS, description="Body point pinned in the frame; scale and rotation from the torso")
    follow_pose:        Field[bool] = Field(True, description="The view follows the pose; off, it is a fixed window of the camera image", newline=True)
    fixed_x:            Field[float] = Field(0.5, min=0.0, max=1.0, description="Fixed window's center across the camera image")
    fixed_y:            Field[float] = Field(0.5, min=0.0, max=1.0, description="Fixed window's center down the camera image")
    fixed_height:       Field[float] = Field(1.0, min=0.05, max=2.0, description="Fixed window's height, as a fraction of the camera image's")


class CenterMaskLayerSettings(BaseSettings):
    """Configuration for CenterMaskLayer temporal blending and blur."""
    blend_factor:   Field[float] = Field(0.2, min=0.0, max=1.0, description="Temporal blending strength (0=static, 1=instant)")
    dilation_steps: Field[int]   = Field(0, min=-10, max=10, description="Mask dilation iterations (positive=expand, negative=shrink)")
    blur_steps:     Field[int]   = Field(0, min=0, max=10, description="Number of blur iterations")
    blur_sigma:     Field[float] = Field(1.7, min=0.0, max=20.0, description="Blur's Gaussian per iteration (pixels)")
    guided_radius:  Field[float] = Field(0.0, min=0.0, max=8.0, step=0.25, newline=True, description="How far the image moves the mask's edge to its own, in the source mask's pixels; 0 = off")
    edge_contrast:  Field[float] = Field(0.03, min=0.001, max=0.3, step=0.001, description="The image's brightness spread (0 to 1) above which the mask's edge follows the image's")


class CenterCamLayerSettings(BaseSettings):
    """Configuration for CenterCamLayer camera rendering."""
    blend_factor:   Field[float] = Field(0.2, min=0.0, max=1.0, description="Camera frame temporal blending")
    mask_opacity:   Field[float] = Field(1.0, min=0.0, max=1.0, description="Mask alpha strength")
    use_mask:       Field[bool]  = Field(True, description="Apply mask to camera output")


class CenterForegroundLayerSettings(BaseSettings):
    """Configuration for CenterForegroundLayer: its colorize."""
    enabled:        Field[bool]  = Field(True, description="Colorize on; off, the foreground as it is")
    blend_factor:   Field[float] = Field(0.2, min=0.0, max=1.0, description="Foreground temporal blending")
    exposure:       Field[float] = Field(1.2, min=0.1, max=3.0, description="Exposure multiplier")
    gamma:          Field[float] = Field(1.0, min=0.5, max=2.0, description="Gamma correction")
    offset:         Field[float] = Field(0.0, min=-0.5, max=0.5, description="Exposure offset")
    contrast:       Field[float] = Field(1.1, min=0.5, max=2.0, description="Contrast")
    saturation:     Field[float] = Field(2.0, min=0.5, max=2.0, description="Color saturation")
    # Cel shade
    levels:         Field[int]   = Field(9, min=0, max=12, description="Number of color bands (0 = off: no bands)")
    smoothness:     Field[float] = Field(0.33, min=0.0, max=0.5, description="Gradient between bands")
    # Colorize
    colorize:       Field[float] = Field(0.96, min=0.0, max=1.0, description="Colorize toward the track color (0.0 = off)")
    # Post
    sharpen:        Field[float] = Field(0.0, min=0.0, max=2.0, description="Sharpen result (0.0 = off)")
    use_mask:       Field[bool]  = Field(True, description="Apply mask to foreground")


# The flow's stage layers' settings, hd_trio's copy (render/layers/flow)
class StageLayerSettings(BaseSettings):
    """What every stage layer's settings hold, first on its panel: the reset. Each stage declares its
    working size itself (`width`, `height`, the 3D fluid's `depth`), after the switches it puts first."""
    reset: Field[bool] =    Field(False, widget=Widget.button, description="Start the stage from rest")


class MotionLayerSettings(StageLayerSettings):
    """What the 2D and the 3D motion layer share; each adds its optical flow's group."""
    guard: Field[bool] =                    Field(True, description="Set non-finite values to 0, count and log them; off once the motion is stable")
    width: Field[int] =                     Field(512, min=32, max=4096, step=32, description="Optical flow processing width")
    height: Field[int] =                    Field(288, min=32, max=4096, step=32, description="Optical flow processing height")
    fps: Field[float] =                     Field(30.0, min=1.0, max=120.0, access=Field.INIT, visible=False, description="Camera frame rate the capture gaps are held around")


class BridgeLayerSettings(StageLayerSettings):
    """What the 2D and the 3D bridge layer share; each adds its bridges' groups."""
    width: Field[int] =     Field(2160, min=32, max=4096, step=32, description="Width the bridge writes its outputs at")
    height: Field[int] =    Field(3840, min=32, max=4096, step=32, description="Height the bridge writes its outputs at")


class FluidLayerSettings(StageLayerSettings):
    """What the 2D and the 3D fluid layer share; each adds its fluid's group, and its `blend_mode` where its
    panel puts the drawing."""


class Motion2DLayerSettings(GradientFlowSettings, MotionLayerSettings):
    """Configuration for Motion2DLayer: the stage's settings and its flow's own, one panel, the stage's first."""
    guard: Field[bool] = Field(False, visible=False, description="The 2D flow has no guard")


class Bridge2DLayerSettings(BridgeLayerSettings):
    """Configuration for Bridge2DLayer: each bridge's group."""
    trail       = Group(bridge_2d.SmoothTrailSettings)
    density     = Group(bridge_2d.DensityBridgeSettings)
    temperature = Group(bridge_2d.TemperatureBridgeSettings)


class Fluid2DLayerSettings(Fluid2DSettings, FluidLayerSettings):
    """Configuration for Fluid2DLayer: the stage's settings and its fluid's own, one panel, the stage's
    first; the stage's width and height are the fluid's simulation grid."""
    blend_mode: Field[BlendState.BlendMode] =    Field(BlendState.BlendMode.ADD, newline=True, description="How the show is drawn over the layers before it")


class LayerGroup(BaseSettings):
    select: Group[LayerSettings]                 = Group(LayerSettings)
    composite: Group[render.LayerCompositorSettings] = Group(render.LayerCompositorSettings)

class DataGroup(BaseSettings):
    a: Group[_DataLayerSettings] = Group(_DataLayerSettings)
    b: Group[_DataLayerSettings] = Group(_DataLayerSettings)

class PreviewGroup(BaseSettings):
    tracker: Group[_TrackerCompositorSettings] = Group(_TrackerCompositorSettings)
    pose   : Group[_PoseCompositorSettings]    = Group(_PoseCompositorSettings)

class CenterGroup(BaseSettings):
    # Received from RenderSettings and passed on to the color mask.
    similarity_scale: Field[float] = Field(1.0, min=0.0, max=1.0, visible=False, description="Cross-camera response")

    geometry: Group[CenterGeometrySettings]                 = Group(CenterGeometrySettings)
    mask    : Group[CenterMaskLayerSettings]                = Group(CenterMaskLayerSettings)
    cam     : Group[CenterCamLayerSettings]                 = Group(CenterCamLayerSettings)
    foreground: Group[CenterForegroundLayerSettings]        = Group(CenterForegroundLayerSettings)
    pose    : Group[render.ViewPoseLayerSettings]           = Group(render.ViewPoseLayerSettings)
    color   : Group[render.SimilaritySilhouetteLayerSettings]        = Group(render.SimilaritySilhouetteLayerSettings, share=[similarity_scale])

class IntroSequenceSettings(BaseSettings):
    """Settings for prerecorded pose overlay during INTRO stages."""
    verbose:           Field[bool]      = Field(False, description="Log start/stop events")
    loop:              Field[bool]      = Field(False, description="Loop recording continuously")
    intro_path:        Field[str]       = Field("intro", access=Field.INIT, description="Path to intro recordings folder")
    refresh_path:      Field[bool]      = Field(False, widget=Widget.button, description="Refresh intro folder list")
    available_folders: Field[list[str]] = Field([""], access=Field.READ, visible=False, description="Available intro recordings")
    folder:            Field[str]       = Field("", widget=Widget.text_select, options=available_folders, description="Selected intro recording")
    source_track:      Field[int]       = Field(0, min=0, max=7, description="Track ID to use from recording", access=Field.READWRITE)

    # -- authoring a new intro recording --
    record_track:      Field[int]       = Field(1, min=0, max=7, description="Camera to record the intro from — set source_track to match when playing it back")
    record_start:      Field[ShowStage] = Field(ShowStage.WHITE_POSE, newline=True, description="Show stage where recording begins")
    record_end:        Field[ShowStage] = Field(ShowStage.ENJOY_IN, description="Last show stage recorded (inclusive)")
    arm_record:        Field[bool]      = Field(False, widget=Widget.toggle, description="Record the next show's window into a new intro folder", access=Field.READWRITE)
    recorder:          Group[_IntroRecorderSettings]     = Group(_IntroRecorderSettings)

    color:             Field[Color]     = Field(Color(1.0, 1.0, 1.0), description="Skeleton color for intro overlay")
    pose:              Group[render.ViewPoseLayerSettings]  = Group(render.ViewPoseLayerSettings)


class RenderSettings(BaseSettings):
    stage:       Field[Stage] = Field(Stage.LERP, description="Pipeline stage for flow/fluid/color layers")
    similarity_scale: Field[float] = Field(1.0, min=0.0, max=1.0, description="Cross-camera response for fluid and color mask (driven per show stage)", access=Field.READWRITE)
    num_cams:    Field[int]   = Field(3, access=Field.INIT, visible=False, description="Number of cameras")
    max_players: Field[int]   = Field(3, access=Field.INIT, visible=False, description="Number of players")
    input_fps:   Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT, visible=False, description="Camera frame rate")
    layer  : Group[LayerGroup]                = Group(LayerGroup)
    data   : Group[DataGroup]                 = Group(DataGroup)
    data_time: Group[_MotionTimeDrawSettings] = Group(_MotionTimeDrawSettings, description="Each player's motion time, as text")
    preview: Group[PreviewGroup]              = Group(PreviewGroup)
    center : Group[CenterGroup]               = Group(CenterGroup, share=[similarity_scale])
    intro_sequence: Group[IntroSequenceSettings] = Group(IntroSequenceSettings)
    motion : Group[Motion2DLayerSettings]     = Group(Motion2DLayerSettings, share=[input_fps.as_('fps')])
    bridge : Group[Bridge2DLayerSettings]     = Group(Bridge2DLayerSettings)
    fluid  : Group[Fluid2DLayerSettings]      = Group(Fluid2DLayerSettings)
    colors : Group[render.ColorSettings]      = Group(render.ColorSettings)


class CadenceSettings(BaseSettings):
    """Pinned: the drops counted where they happen, on the way from the cameras to the poses."""
    frame_sync_drops:    Field[int] = Field(0, access=Field.READ, pinned=True, label="Frame sync drops",
                                            description="Camera frames the frames' sync dropped, its queue full")
    tracklet_sync_drops: Field[int] = Field(0, access=Field.READ, pinned=True, label="Tracklet sync drops",
                                            description="Tracklet messages the tracklets' sync dropped, its queue full")
    upload_drops:        Field[int] = Field(0, access=Field.READ, pinned=True, label="Upload drops",
                                            description="Camera sets the uploader dropped whole, its queue full")
    pose_drops:          Field[int] = Field(0, access=Field.READ, pinned=True, label="Pose drops",
                                            description="Pose model results dropped before the stages after it took them")


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class WindowsSettings(WindowManagerSettings):
    """The render's windows: window 0 the operator's, windows 1-3 the cameras'"""
    window_0: Group[WindowSettings]        = Group(WindowSettings)
    window_1: Group[WindowSettings]        = Group(WindowSettings)
    window_2: Group[WindowSettings]        = Group(WindowSettings)
    window_3: Group[WindowSettings]        = Group(WindowSettings)


class SetupSettings(BaseSettings):
    """The machine and the site: the cameras, the pose, the render, the windows and the links."""
    max_players: Field[int]   = Field(3, access=Field.INIT)
    num_cams   : Field[int]   = Field(3, access=Field.INIT, description="Number of cameras")
    input_fps  : Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)
    render_fps : Field[float] = Field(60.0, access=Field.READWRITE)

    camera : Group[OakGroup]     = Group(OakGroup, share=[num_cams, input_fps.as_('fps')])
    inout  : Group[InOutGroup]   = Group(InOutGroup)
    pose   : Group[PoseGroup]    = Group(PoseGroup, share=[max_players, input_fps.as_('frequency'), render_fps.as_('output_frequency')])
    render : Group[RenderSettings]  = Group(RenderSettings, share=[max_players, num_cams, input_fps])
    windows: Group[WindowsSettings] = Group(WindowsSettings)
    panel : Group[PanelSettings] = Group(PanelSettings)
    session: Group[SessionGroup] = Group(SessionGroup, share=[num_cams, input_fps.as_('fps')])
    cadence: Group[CadenceSettings] = Group(CadenceSettings)


class Settings(BaseSettings):
    setup: Group[SetupSettings] = Group(SetupSettings)
