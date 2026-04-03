"""RenderSettings - reactive settings for the render pipeline."""

from .color_settings import ColorSettings
from .layer_settings import LayerSettings
from . import layers
from modules.settings import Settings, Group
from modules.gl.WindowManager import WindowSettings


class LayerGroup(Settings):
    select = Group(LayerSettings)
    lut    = Group(layers.CompositeLayerSettings)

class DataGroup(Settings):
    a = Group(layers.DataLayerSettings)
    b = Group(layers.DataLayerSettings)

class PreviewGroup(Settings):
    tracker = Group(layers.TrackerCompSettings)
    poser   = Group(layers.PoseCompSettings)

class CentreGroup(Settings):
    geometry = Group(layers.CentreGeomSettings)
    mask     = Group(layers.CentreMaskSettings)
    cam      = Group(layers.CentreCamSettings)
    frg      = Group(layers.CentreFrgSettings)
    pose     = Group(layers.CentrePoseSettings)
    color    = Group(layers.ColorMaskLayerSettings)


class RenderSettings(Settings):
    layer   = Group(LayerGroup)
    data    = Group(DataGroup)
    preview = Group(PreviewGroup)
    centre  = Group(CentreGroup)
    flow    = Group(layers.FlowLayerSettings)
    fluid   = Group(layers.FluidLayerSettings)
    # fluid3D = Group(layers.Fluid3DLayerSettings)
    colors  = Group(ColorSettings)
    window  = Group(WindowSettings)