"""Deep Flow settings — single-camera 3D volumetric fluid installation."""

from enum import IntEnum, auto

from modules.settings import BaseSettings, NiceSettings, Field, Group
from modules.oak import CameraSettings, FrameType, CoderFormat, SimulatorSettings, RecorderSettings, SyncSettings
from modules.render.color_settings import ColorSettings
from modules.render import layers
from modules.inout import OscSoundSettings, OscReceiverSettings
from modules.tracker import OnePerCamTrackerSettings
from modules.pose import batch, nodes, trackers
from modules.pose.batch.model_types import ModelType
from modules.gl.WindowManager import WindowSettings


# ---------------------------------------------------------------------------
#  Layers enum
# ---------------------------------------------------------------------------

class Layers(IntEnum):
    # source layers
    cam_image =     0
    cam_mask =      auto()
    cam_frg =       auto()
    cam_crop =      auto()
    # composite layers
    tracker =       auto()
    poser =         auto()
    # centre layers
    centre_geom =   auto()
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
    fluid3d =       auto()
    composite =     auto()


# ---------------------------------------------------------------------------
#  Layer selection settings (split view)
# ---------------------------------------------------------------------------

class LayerSettings(BaseSettings):
    left:   Field[list[Layers]] = Field([Layers.composite],  description="Layers drawn in the left viewport")
    right:  Field[list[Layers]] = Field([Layers.tracker],    description="Layers drawn in the right viewport")
    final:  Field[list[Layers]] = Field([Layers.composite],  description="Layers drawn on the output monitors")


# ---------------------------------------------------------------------------
#  Oak camera group (1 camera)
# ---------------------------------------------------------------------------

class OakGroup(BaseSettings):
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
    simulator   = Group(SimulatorSettings, share=[video_path, video_format, video_frame_types, num_cameras, fps, color, square, stereo])
    recorder    = Group(RecorderSettings, share=[video_path, temp_path, video_format, video_frame_types, color, square, stereo, num_cameras, fps])
    frame_sync  = Group(SyncSettings, share=[num_cameras, fps])
    tracklet_sync = Group(SyncSettings, share=[num_cameras, fps])

    @property
    def cameras(self) -> list[CameraSettings]:
        return [self.cam_0]


# ---------------------------------------------------------------------------
#  InOut group (OSC sound + OSC control)
# ---------------------------------------------------------------------------

class InOutGroup(BaseSettings):
    osc_sound   = Group(OscSoundSettings)
    osc_control = Group(OscReceiverSettings)


# ---------------------------------------------------------------------------
#  Pose feature groups
# ---------------------------------------------------------------------------

class BboxFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    smoother     = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction   = Group(nodes.PredictorSettings, share=[frequency])
    interpolator = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class PointFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    confidence_filter = Group(nodes.DualConfFilterSettings)
    smoother     = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction   = Group(nodes.PredictorSettings, share=[frequency])
    interpolator = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])

class AngleFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    smoother     = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction   = Group(nodes.PredictorSettings, share=[frequency])
    interpolator = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky       = Group(nodes.StickyFillerSettings)

class VelocityFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0, access=Field.INIT)

    extractor     = Group(nodes.AngleVelExtractorSettings, share=[frequency])
    smoother         = Group(nodes.EuroSmootherSettings, share=[frequency])
    prediction          = Group(nodes.PredictorSettings, share=[frequency])
    interpolator  = Group(nodes.ChaseInterpolatorSettings, share=[frequency.as_('input_frequency'), output_frequency])
    sticky           = Group(nodes.StickyFillerSettings)

class MotionFeature(BaseSettings):
    extractor       = Group(nodes.AngleMotionExtractorSettings)
    moving_average  = Group(nodes.MovingAverageSettings)


# ---------------------------------------------------------------------------
#  Pose pipeline group (no similarity — single player)
# ---------------------------------------------------------------------------

class PoseGroup(BaseSettings):
    max_poses:          Field[int]          = Field(1, min=1, max=16, access=Field.INIT)
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
    motion_gate  = Group(nodes.MotionGateApplicatorSettings, share=[max_poses])
    window_raw   = Group(trackers.WindowNodeSettings)
    window_smooth = Group(trackers.WindowNodeSettings)
    window_lerp  = Group(trackers.WindowNodeSettings)


# ---------------------------------------------------------------------------
#  Tracker
# ---------------------------------------------------------------------------

class TTGroup(BaseSettings):
    tracker = Group(OnePerCamTrackerSettings)


# ---------------------------------------------------------------------------
#  Render settings
# ---------------------------------------------------------------------------

class LayerGroup(BaseSettings):
    select = Group(LayerSettings)
    lut    = Group(layers.CompositeLayerSettings)

class DataGroup(BaseSettings):
    a = Group(layers.DataLayerSettings)
    b = Group(layers.DataLayerSettings)

class PreviewGroup(BaseSettings):
    tracker = Group(layers.TrackerCompSettings)
    poser   = Group(layers.PoseCompSettings)

class CentreGroup(BaseSettings):
    geometry = Group(layers.CentreGeomSettings)
    mask     = Group(layers.CentreMaskSettings)
    cam      = Group(layers.CentreCamSettings)
    frg      = Group(layers.CentreFrgSettings)
    pose     = Group(layers.CentrePoseSettings)
    color    = Group(layers.ColorMaskLayerSettings)

class RenderGroup(BaseSettings):
    layer   = Group(LayerGroup)
    data    = Group(DataGroup)
    preview = Group(PreviewGroup)
    centre  = Group(CentreGroup)
    flow    = Group(layers.FlowLayerSettings)
    fluid3d = Group(layers.Fluid3DLayerSettings)
    colors  = Group(ColorSettings)
    window  = Group(WindowSettings)


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class DeepFlowSettings(BaseSettings):
    num_players:        Field[int]   = Field(1, access=Field.INIT, visible=False)
    fps:                Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT)

    camera  = Group(OakGroup, share=[num_players.as_('num_cameras'), fps])
    tt      = Group(TTGroup)
    pose    = Group(PoseGroup, share=[num_players.as_('max_poses'), fps.as_('frequency')])
    render  = Group(RenderGroup)
    inout   = Group(InOutGroup)
    server  = Group(NiceSettings)
