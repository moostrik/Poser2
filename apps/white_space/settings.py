"""White Space settings — 3-camera panoramic installation with circular LED output.

PRESET MAINTENANCE
------------------
Preset JSON files live in ``apps/white_space/data/settings/``.
Each JSON mirrors this settings tree exactly.  When you rename, add,
or remove a Field here, update every ``.json`` file in that directory
to match — delete stale keys, add new keys with their Field default.
The root class is ``Settings``.
"""

from enum import IntEnum, auto

from modules.settings import BaseSettings, NiceSettings, Field, Group, Widget
from modules.oak import CameraSettings, CameraResolution, MountCheckSettings, SimulatorSettings, RecorderSettings, SyncSettings
from modules.render import layers, ColorSettings
from modules.render.layers import LayerMode
from modules.inout import OscSoundSettings, OscReceiverSettings
from modules.tracker import PanoramicTrackerSettings
from modules.pose import nodes, trackers, window, analytics
from modules import inference
from modules.session import SessionSettings
from modules.gl import WindowSettings
from .light import LightSettings
from .inout import OscLightSenderSettings, UdpLightReceiverSettings
from .pose import GhosterSettings
from .statemachine import StateMachineSettings


# ---------------------------------------------------------------------------
#  Pipeline stages
# ---------------------------------------------------------------------------

class Stage(IntEnum):
    RAW     = 0
    CLEAN   = auto()
    SMOOTH  = auto()
    PREDICT = auto()
    LERP    = auto()


class CameraView(IntEnum):
    """Which of the two camera-derived rows the window shows.

    They show the same pixels in two projections — the frames as each camera delivers them, and
    all four stitched into one 360° strip at the rig centre — so which one you want depends on
    what you are doing: aiming a camera, or reading the calibration. Hiding one gives its height
    to the other and skips its compositing entirely.
    """
    BOTH     = 0
    CAMERAS  = auto()   # the four delivered frames only
    PANORAMA = auto()   # the 360° strip only


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
    cam_panorama = auto()   # the 360° calibration strip: the stitch with the tracker data over it
    ws_light     = auto()   # the ring (fixture in projection mode)
    ws_beam       = auto()   # the bar's four lights (fixture in beam mode); shares ws_light's row
    # data
    data_W       = auto()
    data_F       = auto()
    data_time    = auto()
    data_playhead_W = auto()
    data_playhead_F = auto()


# ---------------------------------------------------------------------------
#  Oak camera group (3 panoramic cameras)
# ---------------------------------------------------------------------------

class OakGroup(BaseSettings):
    fov               : Field[float]           = Field(127.0, access=Field.INIT, description="Camera horizontal FOV (°), quoted for the full sensor width")
    num_cameras       : Field[int]             = Field(4, access=Field.INIT, visible=False, description="Number of cameras")
    fps               : Field[float]           = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    yolo              : Field[bool]            = Field(True, access=Field.INIT, description="Enable YOLO person detection")
    color             : Field[bool]            = Field(False, access=Field.INIT, description="Color camera (False = mono)")
    square            : Field[bool]            = Field(True, access=Field.INIT, description="Use square aspect ratio")
    stereo            : Field[bool]            = Field(False, access=Field.INIT, description="Enable stereo mode")
    resolution        : Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT, description="Sensor mode, all four cameras. P800 is the full readout, P720 a vertical crop")
    frame_height      : Field[int]             = Field(0, access=Field.INIT, step=16, description="Delivered frame height (px, multiple of 16); 0 = derived from the tilt at startup")
    sim_enabled       : Field[bool]            = Field(False, access=Field.INIT, description="Enable simulation mode")
    model_path        : Field[str]             = Field("data/models", access=Field.INIT, description="Model files directory")
    ir_flood_light    : Field[float]           = Field(0.8, min=0.0, max=1.0, widget=Widget.slider, description="IR flood light")
    tilt              : Field[float]           = Field(0.0, access=Field.INIT, description="Camera up-tilt (degrees), positive = aimed upward, the same for all four cameras. Baked into the warp when the devices open.")
    lens_fov          : Field[float]           = Field(0.0, access=Field.INIT, step=0.1, description="Field (°) the lens spans across the full sensor width; 0 = same as fov")
    lens_centre_x     : Field[float]           = Field(0.0, access=Field.INIT, step=0.5, description="Optical centre offset from the frame centre (px), full sensor mode")
    lens_centre_y     : Field[float]           = Field(0.0, access=Field.INIT, step=0.5, description="Optical centre offset from the frame centre (px), positive = down")
    mono_auto_exposure: Field[bool]            = Field(True, widget=Widget.switch, description="Mono auto exposure, all four cameras")

    _cam_share: list = [fps, color, square, stereo, yolo, resolution, frame_height, model_path, ir_flood_light, fov, tilt,
                        lens_fov, lens_centre_x, lens_centre_y, mono_auto_exposure]

    cam_0     : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    cam_1     : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    cam_2     : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    cam_3     : Group[CameraSettings]            = Group(CameraSettings, share=_cam_share)
    mount     : Group[MountCheckSettings]        = Group(MountCheckSettings)
    simulator : Group[SimulatorSettings]         = Group(SimulatorSettings, share=[num_cameras, fps])
    frame_sync: Group[SyncSettings]              = Group(SyncSettings, share=[num_cameras, fps])
    tracklet_sync: Group[SyncSettings]           = Group(SyncSettings, share=[num_cameras, fps])

    @property
    def cameras(self) -> list[CameraSettings]:
        return [self.cam_0, self.cam_1, self.cam_2, self.cam_3]


# ---------------------------------------------------------------------------
#  InOut group (OSC only — no ArtNet, light output via WSPipeline UDP)
# ---------------------------------------------------------------------------

class _OscSoundSettings(OscSoundSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage to read poses from")
    virtual_players: Field[int] = Field(0, min=0, max=16, access=Field.INIT, visible=False, description="Extra virtual (ghost) id slots beyond max_players (shared from root num_virtual)")
    volume: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, newline=True, pinned=True, description="Main sound volume — sent to Max on /global/volume")
    speaker_offset: Field[float] = Field(0.0, min=0.0, max=360.0, step=1.0, description="Where speaker 0 stands, as an azimuth (degrees) — Max adds it in its panner. 0 when the speakers are placed by the layout")


class InOutGroup(BaseSettings):
    """Sender/receiver per domain: the light sender feeds the fixture and its receiver
    hears the fall sensor (plain UDP); the sound sender feeds Max and its receiver hears
    /WS/sound/level (real OSC)."""
    num_players:     Field[int] = Field(8,   access=Field.INIT, visible=False)
    num_virtual:     Field[int] = Field(8,   access=Field.INIT, visible=False)
    resolution:      Field[int] = Field(3600, access=Field.INIT, visible=False)
    osc_light_sender  : Group[OscLightSenderSettings]   = Group(OscLightSenderSettings, share=[resolution])
    udp_light_receiver: Group[UdpLightReceiverSettings] = Group(UdpLightReceiverSettings)
    # OSC sends max_players (live) + virtual_players (ghost) id slots; both shared from root.
    osc_sound_sender  : Group[_OscSoundSettings]        = Group(_OscSoundSettings, share=[num_players.as_('max_players'), num_virtual.as_('virtual_players')])
    osc_sound_receiver: Group[OscReceiverSettings]      = Group(OscReceiverSettings)


# ---------------------------------------------------------------------------
#  Pose feature groups  (same structure as hd_trio — keep all stages)
# ---------------------------------------------------------------------------

class BboxFeature(BaseSettings):
    frequency        : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency : Field[float] = Field(30.0)


class PointFeature(BaseSettings):
    frequency       : Field[float] = Field(30.0, access=Field.INIT)
    output_frequency: Field[float] = Field(30.0)

    confidence      : Group[nodes.DualConfFilterSettings]    = Group(nodes.DualConfFilterSettings)
    sticky          : Group[nodes.StickyFillerSettings]      = Group(nodes.StickyFillerSettings)


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
    max_poses        : Field[int]       = Field(3, min=1, max=16, access=Field.INIT)
    ghost_slots      : Field[int]       = Field(8, min=0, max=16, access=Field.INIT, visible=False, description="Ghost id pool size (shared from root num_virtual)")
    model_type       : Field[inference.ModelType] = Field(inference.ModelType.TRT, access=Field.INIT)
    model_path       : Field[str]       = Field("", access=Field.INIT, visible=False)
    verbose          : Field[bool]      = Field(False, access=Field.INIT)
    frequency        : Field[float]     = Field(30.0, access=Field.INIT)
    output_frequency : Field[float]     = Field(30.0)
    ws_input_stage   : Field[Stage]     = Field(Stage.LERP, description="Pipeline stage that feeds the WS light pipeline")

    _feature_share: list = [frequency, output_frequency]

    pose            : Group[inference.pose.Settings]         = Group(inference.pose.Settings, share=[max_poses, model_type, model_path, verbose])
    segmentation    : Group[inference.segmentation.Settings] = Group(inference.segmentation.Settings, share=[max_poses, model_type, model_path, verbose])
    image_crop      : Group[inference.crop.Settings]         = Group(inference.crop.Settings, share=[max_poses])
    angle_extractor : Group[nodes.AngleExtractorSettings]    = Group(nodes.AngleExtractorSettings)
    distance_extractor: Group[nodes.DistanceExtractorSettings] = Group(nodes.DistanceExtractorSettings)
    leg_deviation_extractor: Group[nodes.LegDeviationExtractorSettings] = Group(nodes.LegDeviationExtractorSettings)
    torso_tilt_extractor: Group[nodes.TorsoTiltExtractorSettings] = Group(nodes.TorsoTiltExtractorSettings)
    bbox            : Group[BboxFeature]                     = Group(BboxFeature, share=_feature_share)
    point           : Group[PointFeature]                    = Group(PointFeature, share=_feature_share)
    angle           : Group[AngleFeature]                    = Group(AngleFeature, share=_feature_share)
    velocity        : Group[VelocityFeature]                 = Group(VelocityFeature, share=_feature_share)
    motion          : Group[MotionFeature]                   = Group(MotionFeature)
    similarity      : Group[SimilarityFeature]               = Group(SimilarityFeature, share=[frequency, output_frequency, max_poses])
    window_raw      : Group[window.WindowNodeSettings]       = Group(window.WindowNodeSettings)
    window_clean    : Group[window.WindowNodeSettings]       = Group(window.WindowNodeSettings)
    window_smooth   : Group[window.WindowNodeSettings]       = Group(window.WindowNodeSettings)
    window_predict  : Group[window.WindowNodeSettings]       = Group(window.WindowNodeSettings)
    window_lerp     : Group[window.WindowNodeSettings]       = Group(window.WindowNodeSettings)
    # The ghost subsystem — virtual poses injected into the pipeline (feeds the OSC sound
    # id slots and the beam_haunted debug visual; the show's beam_flash is independent of it).
    ghoster         : Group[GhosterSettings]                 = Group(GhosterSettings, share=[max_poses.as_('live_players'), ghost_slots.as_('ghost_slots')])


# ---------------------------------------------------------------------------
#  Recording group (recording lifecycle — independent of the show state machine)
# ---------------------------------------------------------------------------

class RecordingGroup(BaseSettings):
    """App recording group — composes SessionSettings with app-specific recorders.
    Recording is decoupled from the show: it works stand-alone and during session mode,
    and nothing here touches the state machine."""
    num_cameras:   Field[int]   = Field(4, access=Field.INIT, description="Number of cameras")
    fps:           Field[float] = Field(30.0, access=Field.INIT, description="Camera frame rate")

    start:         Field[bool]  = Field(False, widget=Widget.button, description="Start recording")
    stop:          Field[bool]  = Field(False, widget=Widget.button, description="Stop recording")
    output_path:   Field[str]   = Field("recordings", access=Field.INIT, description="Recordings output directory")
    name:          Field[str]   = Field("", widget=Widget.input, description="Recording group ID")
    split:         Field[bool]  = Field(False, widget=Widget.button, description="Split chunk", visible=False)
    split_seconds: Field[float] = Field(10, min=1, max=60, widget=Widget.number, description="Split recording into chunks (seconds)")

    _session_share:  list = [output_path, name, start, stop, split, split_seconds]
    _recorder_share: list = [start, stop, split, name, output_path]

    osc:      Group[OscReceiverSettings]    = Group(OscReceiverSettings)
    core:     Group[SessionSettings]        = Group(SessionSettings, share=_session_share)
    video:    Group[RecorderSettings]       = Group(RecorderSettings, share=_recorder_share + [num_cameras, fps])


# ---------------------------------------------------------------------------
#  Render settings
# ---------------------------------------------------------------------------

class _TrackerCompSettings(layers.TrackerCompSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for pose data")

class _PoseCompSettings(layers.PoseCompSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for camera crop")

class _MTimeSettings(layers.MTimeRendererSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for pose data")

class _DataLayerSettings(layers.DataLayerSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for pose data")


class PreviewGroup(BaseSettings):
    tracker: Group[_TrackerCompSettings] = Group(_TrackerCompSettings)
    poser  : Group[_PoseCompSettings]    = Group(_PoseCompSettings)


class PlayheadFeatureSelect(IntEnum):
    """App-local feature dropdown for the playhead data layers (keys PLAYHEAD_FEATURE_MAP)."""
    PlayheadOffset    = 0
    GhostFeature      = auto()


class PlayheadDataLayerSettings(BaseSettings):
    """App-owned data-layer config (mirrors DataLayerSettings, but its own feature dropdown).

    Drives the generic FeatureWindowLayer/FeatureFrameLayer over the app's playhead features.
    Defaults off (mode NONE) and to the LERP stage — the only stage where the features exist.
    Set the generic ``data.mode`` to NONE when enabling this to avoid two graphs overlapping.
    """
    stage:             Field[Stage]               = Field(Stage.LERP)
    mode:              Field[LayerMode]            = Field(LayerMode.NONE)
    feature_field:     Field[PlayheadFeatureSelect] = Field(PlayheadFeatureSelect.GhostFeature)
    line_width:        Field[float]               = Field(3.0)
    line_smooth:       Field[float]               = Field(1.0)
    use_scores:        Field[bool]                = Field(False)
    render_labels:     Field[bool]                = Field(True)
    use_history_color: Field[bool]                = Field(False)


class BeamLightSimSettings(BaseSettings):
    """The render's simulation of the bar in beam mode: the four beam lights as lines on the
    walls."""
    width: Field[float] = Field(15.0, min=0.0, max=90.0, step=0.5, description="Solid width of a beam light's line on the wall (deg)")
    blur: Field[float] = Field(6.0,  min=0.0, max=45.0, step=0.5, description="Soft falloff on each side of the line (deg); 0 = a hard edge")


class RenderSettings(BaseSettings):
    camera_view: Field[CameraView] = Field(CameraView.BOTH, widget=Widget.select,
                                           description="Which camera row to show: the delivered frames, the 360° strip, or both")
    num_cams:    Field[int]  = Field(4, access=Field.INIT, visible=False, description="Number of cameras")
    num_players: Field[int]  = Field(4, access=Field.INIT, visible=False, description="Number of players")
    tilt:        Field[float] = Field(0.0, access=Field.INIT, visible=False, description="Camera up-tilt (°), shared from the root — relayed to the panorama layer")
    resolution:  Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT, visible=False, description="Sensor mode, shared from the root — the camera row's aspect follows it")
    frame_height: Field[int] = Field(0, access=Field.INIT, visible=False, description="Delivered frame height, shared from the root — the camera row's aspect follows it")
    preview:     Group[PreviewGroup]        = Group(PreviewGroup)
    data_time:   Group[_MTimeSettings]      = Group(_MTimeSettings)
    data:        Group[_DataLayerSettings]  = Group(_DataLayerSettings)
    playhead_data: Group[PlayheadDataLayerSettings] = Group(PlayheadDataLayerSettings)
    beam_light_sim: Group[BeamLightSimSettings] = Group(BeamLightSimSettings)
    # Shared down from the root so the panorama can size itself to the picture the mount
    # actually delivers; it moves nothing, the warp has already levelled the frame.
    panorama:    Group[layers.PanoramaLayerSettings] = Group(layers.PanoramaLayerSettings, share=[tilt])
    colors:      Group[ColorSettings]       = Group(ColorSettings)
    window:      Group[WindowSettings]      = Group(WindowSettings)


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    num_players     : Field[int]   = Field(4, access=Field.INIT)
    num_virtual     : Field[int]   = Field(8, access=Field.INIT)
    num_cameras     : Field[int]   = Field(4, access=Field.INIT)
    input_fps       : Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)
    render_fps      : Field[float] = Field(30.0)
    light_resolution: Field[int]   = Field(300, min=10, max=1000, access=Field.INIT, description="LED strip resolution (pixels)")
    fov             : Field[float] = Field(127.0, access=Field.INIT, description="Azimuth span (°) of each delivered camera frame — the tracker's contract, baked in at open")
    resolution      : Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT, description="Sensor mode for all cameras")
    frame_height    : Field[int]   = Field(0, access=Field.INIT, step=16, description="Delivered frame height (px, multiple of 16); 0 = derived: the sensor's full reach at this tilt")
    tilt            : Field[float] = Field(0.0, access=Field.INIT, description="Camera up-tilt (°), positive = aimed up, the same for all four")
    # The lens, shared by all four — read off their calibrations; see CALIBRATION.md, Camera.
    lens_fov        : Field[float] = Field(0.0, access=Field.INIT, step=0.1, description="Field (°) the lens spans across the full sensor width; 0 = same as fov")
    lens_centre_x   : Field[float] = Field(0.0, access=Field.INIT, step=0.5, description="Optical centre offset from the frame centre (px), full sensor mode")
    lens_centre_y   : Field[float] = Field(0.0, access=Field.INIT, step=0.5, description="Optical centre offset from the frame centre (px), positive = down")
    spin_down_seconds: Field[float] = Field(10.0, min=1.0, max=60.0, step=0.5, visible=False, description="S9/S10 wall-fade seconds — canonical value tying states (the visible slider) to the wind_down layer")

    # In panel order: what you look at, what you capture, the wires out, the sensors, what is made of
    # them, and what the show does with it.
    render : Group[RenderSettings]  = Group(RenderSettings, share=[num_players, num_cameras.as_('num_cams'), tilt, resolution, frame_height])
    record : Group[RecordingGroup]  = Group(RecordingGroup, share=[num_cameras.as_('num_cameras'), input_fps.as_('fps')])
    inout  : Group[InOutGroup]      = Group(InOutGroup, share=[num_players.as_('num_players'), num_virtual.as_('num_virtual'), light_resolution.as_('resolution')])
    camera : Group[OakGroup]        = Group(OakGroup, share=[num_cameras.as_('num_cameras'), input_fps.as_('fps'), fov, tilt, resolution, frame_height,
                                                             lens_fov, lens_centre_x, lens_centre_y])
    # Its own group, not under `camera`: the tracker fuses the cameras' output, as `pose` does, and
    # nothing in it is a camera setting. The frame's shape is shared in straight from the root.
    track  : Group[PanoramicTrackerSettings] = Group(PanoramicTrackerSettings, share=[fov, resolution, frame_height, tilt,
                                                                                      lens_fov, lens_centre_x, lens_centre_y])
    pose   : Group[PoseGroup]       = Group(PoseGroup, share=[num_players.as_('max_poses'), num_virtual.as_('ghost_slots'), input_fps.as_('frequency'), render_fps.as_('output_frequency')])
    light  : Group[LightSettings]   = Group(LightSettings, share=[num_players.as_('max_poses'), num_cameras.as_('num_cameras'), light_resolution.as_('light_resolution'), fov, spin_down_seconds])
    states : Group[StateMachineSettings] = Group(StateMachineSettings, share=[spin_down_seconds])
    server : Group[NiceSettings]    = Group(NiceSettings)
