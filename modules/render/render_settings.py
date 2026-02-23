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

from modules.render.layers.generic.CompositeLayer import CompositeLayerSettings
from modules.render.layers.generic.MSColorMaskLayer import MSColorMaskLayerSettings
from modules.render.layers.flow.FlowLayer import FlowLayerSettings
from modules.render.layers.flow.FluidLayer import FluidLayerSettings

from modules.render.layers.centre.CentreGeometry import CentreGeometrySettings
from modules.render.layers.centre.CentreMaskLayer import CentreMaskSettings
from modules.render.layers.centre.CentreCamLayer import CentreCamSettings
from modules.render.layers.centre.CentreFrgLayer import CentreFrgSettings
from modules.render.layers.centre.CentrePoseLayer import CentrePoseSettings

from modules.render.layers.cam.TrackerCompositor import TrackerCompSettings
from modules.render.layers.cam.PoseCompositor import PoseCompSettings

from modules.render.layers.data.DataLayerSettings import DataLayerSettings
from modules.render.layers.data.MTimeRenderer import MTimeRendererSettings

from modules.settings import Setting, Child, BaseSettings
from modules.gl.WindowManager import WindowSettings
from modules.render.color_settings import ColorSettings


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
    _config: DataLayerSettings

    def set_active(self, active: bool) -> None: ...


# ---------------------------------------------------------------------------
# Settings classes
# ---------------------------------------------------------------------------


class DataLayerControl(BaseSettings):
    """Data layer control — feature selection, display mode, stages."""
    feature: Setting[RenderFeature] =   Setting(RenderFeature.angle_motion)
    mode: Setting[LayerMode] =      Setting(LayerMode.WINDOW)
    stage_a: Setting[Stage] =       Setting(Stage.SMOOTH)
    stage_b: Setting[Stage] =       Setting(Stage.LERP)


class RenderSettings(BaseSettings):
    """Mutable render-pipeline settings (registered in SettingsRegistry)."""

    # Window / init-only (foldable child)
    window: Child[WindowSettings] =     Child(WindowSettings)

    # Colors
    colors: Child[ColorSettings] =      Child(ColorSettings)

    # Data layer control (foldable child)
    data_layer: Child[DataLayerControl] =  Child(DataLayerControl)

    # Centre layers
    centre_geometry: Child[CentreGeometrySettings] = Child(CentreGeometrySettings)
    centre_mask:     Child[CentreMaskSettings]     = Child(CentreMaskSettings)
    centre_cam:      Child[CentreCamSettings]       = Child(CentreCamSettings)
    centre_frg:      Child[CentreFrgSettings]       = Child(CentreFrgSettings)
    centre_pose:     Child[CentrePoseSettings]      = Child(CentrePoseSettings)

    # Cam compositors
    tracker:         Child[TrackerCompSettings] = Child(TrackerCompSettings)
    pose_comp:       Child[PoseCompSettings]    = Child(PoseCompSettings)

    # Data layers
    data_a:          Child[DataLayerSettings]      = Child(DataLayerSettings)
    data_b:          Child[DataLayerSettings]      = Child(DataLayerSettings)
    data_time:       Child[MTimeRendererSettings]  = Child(MTimeRendererSettings)

    # Composite / mask
    composite:       Child[CompositeLayerSettings]     = Child(CompositeLayerSettings)
    ms_mask:         Child[MSColorMaskLayerSettings]   = Child(MSColorMaskLayerSettings)

    # Flow / Fluid (child configs, shared across all cameras)
    flow: Child[FlowLayerSettings] =          Child(FlowLayerSettings)
    fluid: Child[FluidLayerSettings] =         Child(FluidLayerSettings)

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
        for field in (DataLayerControl.feature, DataLayerControl.mode, DataLayerControl.stage_a, DataLayerControl.stage_b):
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
