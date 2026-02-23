"""RenderSettings — reactive settings for the render pipeline.

Manages mutable runtime state (feature selection, data layer mode, stages,
LUT, flow/fluid configs) via the BaseSettings descriptor system. Registered
in the SettingsRegistry for NiceGUI panel and JSON preset persistence.

Window/init fields (title, width, fps, …) are init_only Settings.
"""

from modules.render.layers.generic.CompositeLayer import CompositeLayerSettings
from modules.render.layers.generic.MSColorMaskLayer import ColorMaskLayerSettings
from modules.render.layers.flow.FlowLayer import FlowLayerSettings
from modules.render.layers.flow.FluidLayer import FluidLayerSettings

from modules.render.layers.centre.CentreGeometry import CentreGeomSettings
from modules.render.layers.centre.CentreMaskLayer import CentreMaskSettings
from modules.render.layers.centre.CentreCamLayer import CentreCamSettings
from modules.render.layers.centre.CentreFrgLayer import CentreFrgSettings
from modules.render.layers.centre.CentrePoseLayer import CentrePoseSettings

from modules.render.layers.cam.TrackerCompositor import TrackerCompSettings
from modules.render.layers.cam.PoseCompositor import PoseCompSettings

from modules.render.layers.data.DataLayerSettings import DataLayerSettings

from modules.settings import Child, BaseSettings
from modules.gl.WindowManager import WindowSettings
from modules.render.color_settings import ColorSettings
from modules.render.layer_ids import LayerSettings


class RenderSettings(BaseSettings):
    """Mutable render-pipeline settings (registered in SettingsRegistry)."""

    # Layer Selection
    layers:          Child[LayerSettings] =         Child(LayerSettings)

    # Data layers
    data_a:          Child[DataLayerSettings] =     Child(DataLayerSettings)
    data_b:          Child[DataLayerSettings] =     Child(DataLayerSettings)

    # Cam compositors
    tracker:        Child[TrackerCompSettings] =    Child(TrackerCompSettings)
    poser:          Child[PoseCompSettings] =       Child(PoseCompSettings)

    # Centre layers
    centre_geometry:Child[CentreGeomSettings] =     Child(CentreGeomSettings)
    centre_mask:    Child[CentreMaskSettings] =     Child(CentreMaskSettings)
    centre_cam:     Child[CentreCamSettings] =      Child(CentreCamSettings)
    centre_frg:     Child[CentreFrgSettings] =      Child(CentreFrgSettings)
    centre_pose:    Child[CentrePoseSettings] =     Child(CentrePoseSettings)

    # Mask
    color_masks:    Child[ColorMaskLayerSettings] =  Child(ColorMaskLayerSettings)

    # Flow / Fluid
    flow:           Child[FlowLayerSettings] =      Child(FlowLayerSettings)
    fluid:          Child[FluidLayerSettings] =     Child(FluidLayerSettings)

    # Lut
    composite:      Child[CompositeLayerSettings] = Child(CompositeLayerSettings)

    # colors
    colors:         Child[ColorSettings] =          Child(ColorSettings)

    # Window
    window:         Child[WindowSettings] =         Child(WindowSettings)
