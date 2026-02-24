"""RenderSettings - reactive settings for the render pipeline."""

from .color_settings import ColorSettings
from .layer_settings import LayerSettings
from . import layers
from modules.settings import Settings
from modules.gl.WindowManager import WindowSettings


class LayerGroup(Settings):
    select:   LayerSettings
    lut:      layers.CompositeLayerSettings

class DataGroup(Settings):
    a:        layers.DataLayerSettings
    b:        layers.DataLayerSettings

class PreviewGroup(Settings):
    tracker:  layers.TrackerCompSettings
    poser:    layers.PoseCompSettings

class CentreGroup(Settings):
    geometry: layers.CentreGeomSettings
    mask:     layers.CentreMaskSettings
    cam:      layers.CentreCamSettings
    frg:      layers.CentreFrgSettings
    pose:     layers.CentrePoseSettings
    color:    layers.ColorMaskLayerSettings


class RenderSettings(Settings):
    layer:   LayerGroup
    data:    DataGroup
    preview: PreviewGroup
    centre:  CentreGroup
    flow:    layers.FlowLayerSettings
    fluid:   layers.FluidLayerSettings
    colors:  ColorSettings
    window:  WindowSettings