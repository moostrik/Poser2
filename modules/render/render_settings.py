"""RenderSettings — reactive settings for the render pipeline.

Manages mutable runtime state (feature selection, data layer mode, stages,
LUT, flow/fluid configs) via the BaseSettings descriptor system.  Declared
as a child of the app-level root settings for NiceGUI panel and JSON preset
persistence.

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

from modules.settings import BaseSettings
from modules.gl.WindowManager import WindowSettings
from modules.render.color_settings import ColorSettings
from modules.render.layer_ids import LayerSettings


class RenderSettings(BaseSettings):
    """Mutable render-pipeline settings (registered in SettingsRegistry)."""

    # Layer Selection
    layers:          LayerSettings

    # Data layers
    data_a:          DataLayerSettings
    data_b:          DataLayerSettings

    # Cam compositors
    tracker:         TrackerCompSettings
    poser:           PoseCompSettings

    # Centre layers
    centre_geometry: CentreGeomSettings
    centre_mask:     CentreMaskSettings
    centre_cam:      CentreCamSettings
    centre_frg:      CentreFrgSettings
    centre_pose:     CentrePoseSettings

    # Mask
    color_masks:     ColorMaskLayerSettings

    # Flow / Fluid
    flow:            FlowLayerSettings
    fluid:           FluidLayerSettings

    # Lut
    composite:       CompositeLayerSettings

    # colors
    colors:          ColorSettings

    # Window
    window:          WindowSettings
