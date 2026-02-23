"""RenderSettings — reactive settings for the render pipeline.

Manages mutable runtime state (feature selection, data layer mode, stages,
LUT, flow/fluid configs) via the BaseSettings descriptor system. Registered
in the SettingsRegistry for NiceGUI panel and JSON preset persistence.

Window/init fields (title, width, fps, …) are init_only Settings.
"""

from enum import IntEnum, auto
from typing import Protocol

from modules.DataHub import Stage
from modules.pose.Frame import ScalarFrameField

from modules.render.layers.generic.CompositeLayer import CompositeLayerConfig
from modules.render.layers.generic.MSColorMaskLayer import MSColorMaskLayerConfig
from modules.render.layers.flow.FlowLayer import FlowLayerConfig
from modules.render.layers.flow.FluidLayer import FluidLayerConfig

from modules.render.layers.centre.CentreGeometry import CentreGeometryConfig
from modules.render.layers.centre.CentreMaskLayer import CentreMaskConfig
from modules.render.layers.centre.CentreCamLayer import CentreCamConfig
from modules.render.layers.centre.CentreFrgLayer import CentreFrgConfig
from modules.render.layers.centre.CentrePoseLayer import CentrePoseConfig

from modules.render.layers.cam.TrackerCompositor import TrackerCompConfig
from modules.render.layers.cam.PoseCompositor import PoseCompConfig

from modules.render.layers.data.DataLayerConfig import DataLayerConfig
from modules.render.layers.data.MTimeRenderer import MTimeRendererConfig

from modules.settings import Setting, Child, BaseSettings
from modules.gl.WindowManager import WindowSettings


# ---------------------------------------------------------------------------
# Enums & protocols
# ---------------------------------------------------------------------------

class RenderFeature(IntEnum):
    """Feature field selection for rendering. NONE disables the feature."""
    NONE =          -1
    bbox =          ScalarFrameField.bbox
    angles =        ScalarFrameField.angles
    angle_vel =     ScalarFrameField.angle_vel
    angle_motion =  ScalarFrameField.angle_motion
    angle_sym =     ScalarFrameField.angle_sym
    similarity =    ScalarFrameField.similarity
    leader =        ScalarFrameField.leader
    motion_gate =   ScalarFrameField.motion_gate


class LayerMode(IntEnum):
    """Display mode for a data layer slot."""
    FRAME =     0
    WINDOW =    auto()


class DataLayer(Protocol):
    """Protocol for data layer instances with active state and shared config."""
    _config: DataLayerConfig

    def set_active(self, active: bool) -> None: ...


# ---------------------------------------------------------------------------
# Settings classes
# ---------------------------------------------------------------------------


class DataLayerSettings(BaseSettings):
    """Data layer control — feature selection, display mode, stages."""
    feature: Setting[RenderFeature] =   Setting(RenderFeature, RenderFeature.angle_motion)
    mode: Setting[LayerMode] =      Setting(LayerMode, LayerMode.WINDOW)
    stage_a: Setting[Stage] =       Setting(Stage, Stage.SMOOTH)
    stage_b: Setting[Stage] =       Setting(Stage, Stage.LERP)


class RenderSettings(BaseSettings):
    """Mutable render-pipeline settings (registered in SettingsRegistry)."""

    # Window / init-only (foldable child)
    window: Child[WindowSettings] =     Child(WindowSettings)

    # Data layer control (foldable child)
    data_layer: Child[DataLayerSettings] =  Child(DataLayerSettings)

    # Centre layers
    centre_geometry: Child[CentreGeometryConfig] = Child(CentreGeometryConfig)
    centre_mask:     Child[CentreMaskConfig]     = Child(CentreMaskConfig)
    centre_cam:      Child[CentreCamConfig]       = Child(CentreCamConfig)
    centre_frg:      Child[CentreFrgConfig]       = Child(CentreFrgConfig)
    centre_pose:     Child[CentrePoseConfig]      = Child(CentrePoseConfig)

    # Cam compositors
    tracker:         Child[TrackerCompConfig] = Child(TrackerCompConfig)
    pose_comp:       Child[PoseCompConfig]    = Child(PoseCompConfig)

    # Data layers
    data_a:          Child[DataLayerConfig]      = Child(DataLayerConfig)
    data_b:          Child[DataLayerConfig]      = Child(DataLayerConfig)
    data_time:       Child[MTimeRendererConfig]  = Child(MTimeRendererConfig)

    # Composite / mask
    composite:       Child[CompositeLayerConfig]     = Child(CompositeLayerConfig)
    ms_mask:         Child[MSColorMaskLayerConfig]   = Child(MSColorMaskLayerConfig)

    # Flow / Fluid (child configs, shared across all cameras)
    flow: Child[FlowLayerConfig] =          Child(FlowLayerConfig)
    fluid: Child[FluidLayerConfig] =         Child(FluidLayerConfig)

    # ------------------------------------------------------------------
    # Data-layer binding (replaces Config.bind / _propagate)
    # ------------------------------------------------------------------

    def bind_data_layers(self,
             windows_a: dict[int, DataLayer], frames_a: dict[int, DataLayer], angvel_a: dict[int, DataLayer],
             windows_b: dict[int, DataLayer], frames_b: dict[int, DataLayer], angvel_b: dict[int, DataLayer]) -> None:
        """Bind data-layer instances so config changes propagate active state."""
        object.__setattr__(self, '_slots', [
            (windows_a, frames_a, angvel_a, 'stage_a'),
            (windows_b, frames_b, angvel_b, 'stage_b'),
        ])
        # Wire _propagate to each relevant field on the data_layer child
        dl = self.data_layer
        for field in (DataLayerSettings.feature, DataLayerSettings.mode, DataLayerSettings.stage_a, DataLayerSettings.stage_b):
            dl.bind(field, lambda _v: self._propagate())
        self._propagate()

    def _propagate(self) -> None:
        """Push current config to bound data layers."""
        dl = self.data_layer
        ff = dl.feature
        scalar_field = None if ff == RenderFeature.NONE else ScalarFrameField(ff.value)
        is_angles = (scalar_field == ScalarFrameField.angles) if scalar_field else False

        for windows, frames, angvel, stage_attr in self._slots:
            stage: Stage = getattr(dl, stage_attr)
            for i in windows:
                # Deactivate all three layer types
                windows[i].set_active(False)
                frames[i].set_active(False)
                angvel[i].set_active(False)

                # Activate and configure the selected layer
                if scalar_field is not None:
                    layer = windows[i] if dl.mode == LayerMode.WINDOW else (angvel[i] if is_angles else frames[i])
                    layer.set_active(True)

                    cfg = layer._config
                    cfg.feature_field = scalar_field
                    cfg.stage = stage
