"""The OAK4 test app's settings: one OAK4, the pose model on the device, the pose pipeline on its keypoints.

PRESET MAINTENANCE
------------------
Preset JSON files live in ``apps/oak4_test/data/settings/``.
Each JSON mirrors this settings tree exactly.  When you rename, add,
or remove a Field here, update every ``.json`` file in that directory
to match — delete stale keys, add new keys with their Field default.
The root class is ``Settings``.
"""

from enum import IntEnum, auto

from modules.settings import BaseSettings, Field, Group, Widget
from modules.panel import PanelSettings
from modules.camera import CameraCheckSettings, CameraResolution, SyncSettings
from modules.camera.oak4 import Oak4CameraSettings
from modules.render import layers, ColorSettings
from modules.tracker import OnePerCamTrackerSettings, PosesFromTrackletsSettings
from modules.pose import nodes, window
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
#  Layers
# ---------------------------------------------------------------------------

class Layers(IntEnum):
    cam_image =     0
    tracker =       auto()   # the camera image with the tracklets, the boxes and the skeletons
    cam_readings =  auto()   # the camera's frame rate, tilt and roll error; draw it over tracker


class LayerSettings(BaseSettings):
    final: Field[list[Layers]] = Field([Layers.tracker, Layers.cam_readings], description="Layers drawn in the window", widget=Widget.playlist)


# ---------------------------------------------------------------------------
#  OAK4 camera group (1 camera)
# ---------------------------------------------------------------------------

class Oak4Group(BaseSettings):
    num_cams:       Field[int]               = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    fps:            Field[float]             = Field(30.0, min=1.0, max=60.0, access=Field.INIT, description="Camera frame rate")
    color:          Field[bool]              = Field(True, access=Field.INIT, description="Color frames (False = grey off the same sensor)")
    resolution:     Field[CameraResolution]  = Field(CameraResolution.P800, access=Field.INIT, description="Delivered size before the crop")
    model_path:     Field[str]               = Field("data/models", access=Field.INIT, visible=False, description="Model files directory")
    fov:            Field[float]             = Field(127.0, access=Field.INIT, description="Camera horizontal FOV over the full sensor width (°)")

    _cam_share: list = [fps, color, resolution, model_path, fov]
    cam_0        : Group[Oak4CameraSettings]       = Group(Oak4CameraSettings, share=_cam_share)
    camera_check : Group[CameraCheckSettings]      = Group(CameraCheckSettings)
    tracker      : Group[OnePerCamTrackerSettings] = Group(OnePerCamTrackerSettings)
    frame_sync   : Group[SyncSettings]             = Group(SyncSettings, share=[num_cams, fps])
    tracklet_sync: Group[SyncSettings]             = Group(SyncSettings, share=[num_cams, fps])

    @property
    def cameras(self) -> list[Oak4CameraSettings]:
        return [self.cam_0]


# ---------------------------------------------------------------------------
#  Pose feature groups
# ---------------------------------------------------------------------------

class BboxFeature(BaseSettings):
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    smoother    : Group[nodes.EuroSmootherSettings]        = Group(nodes.EuroSmootherSettings)
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


class PoseGroup(BaseSettings):
    max_players:        Field[int]   = Field(4, min=1, max=16, access=Field.INIT)
    frequency:          Field[float] = Field(30.0, access=Field.INIT)
    output_frequency:   Field[float] = Field(60.0)

    _feature_share: list = [frequency, output_frequency]

    tracklets      : Group[PosesFromTrackletsSettings]   = Group(PosesFromTrackletsSettings)
    angle_extractor: Group[nodes.AngleExtractorSettings] = Group(nodes.AngleExtractorSettings)
    bbox           : Group[BboxFeature]                  = Group(BboxFeature, share=_feature_share)
    point          : Group[PointFeature]                 = Group(PointFeature, share=_feature_share)
    angle          : Group[AngleFeature]                 = Group(AngleFeature, share=_feature_share)
    window_raw     : Group[window.WindowNodeSettings]    = Group(window.WindowNodeSettings)
    window_clean   : Group[window.WindowNodeSettings]    = Group(window.WindowNodeSettings)
    window_smooth  : Group[window.WindowNodeSettings]    = Group(window.WindowNodeSettings)
    window_lerp    : Group[window.WindowNodeSettings]    = Group(window.WindowNodeSettings)


# ---------------------------------------------------------------------------
#  Render settings
# ---------------------------------------------------------------------------

class _TrackerCompositorSettings(layers.TrackerCompositorSettings):
    stage: Field[Stage] = Field(Stage.LERP, description="Pipeline stage for pose data")


class RenderSettings(BaseSettings):
    layer  : Group[LayerSettings]              = Group(LayerSettings)
    tracker: Group[_TrackerCompositorSettings] = Group(_TrackerCompositorSettings)
    colors : Group[ColorSettings]              = Group(ColorSettings)
    window : Group[WindowSettings]             = Group(WindowSettings)


# ---------------------------------------------------------------------------
#  Root settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    max_players:        Field[int]   = Field(4, access=Field.INIT, visible=False)
    num_cams:           Field[int]   = Field(1, access=Field.INIT, visible=False, description="Number of cameras")
    input_fps:          Field[float] = Field(30.0, min=1.0, max=60.0, access=Field.INIT)

    camera: Group[Oak4Group]      = Group(Oak4Group, share=[num_cams, input_fps.as_('fps')])
    pose  : Group[PoseGroup]      = Group(PoseGroup, share=[max_players, input_fps.as_('frequency')])
    render: Group[RenderSettings] = Group(RenderSettings)
    panel : Group[PanelSettings]  = Group(PanelSettings)
