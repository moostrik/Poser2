"""RenderSettings - reactive settings for the render pipeline."""

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


class LayerGroup(BaseSettings):
    select: LayerSettings
    lut:    CompositeLayerSettings


class DataGroup(BaseSettings):
    a: DataLayerSettings
    b: DataLayerSettings


class PreviewGroup(BaseSettings):
    tracker: TrackerCompSettings
    poser:   PoseCompSettings


class CentreGroup(BaseSettings):
    geometry: CentreGeomSettings
    mask:     CentreMaskSettings
    cam:      CentreCamSettings
    frg:      CentreFrgSettings
    pose:     CentrePoseSettings
    color:    ColorMaskLayerSettings


class RenderSettings(BaseSettings):
    """Mutable render-pipeline settings."""
    layer:   LayerGroup
    data:    DataGroup
    preview: PreviewGroup
    centre:  CentreGroup
    flow:    FlowLayerSettings
    fluid:   FluidLayerSettings
    colors:  ColorSettings
    window:  WindowSettings