"""Layer identifiers for the render pipeline."""

from enum import IntEnum, auto

from modules.settings import Setting, BaseSettings


class Layers(IntEnum):

    # source layers
    cam_image =     0
    cam_mask =      auto()
    cam_frg=        auto()
    cam_crop =      auto()

    # composite layers
    tracker =       auto()
    poser =         auto()

    # centre layers
    centre_geom=    auto()
    centre_cam =    auto()
    centre_mask =   auto()
    centre_frg =    auto()
    centre_pose =   auto()

    # Data layers
    data_B_W =      auto()
    data_B_F =      auto()
    data_A_W =      auto()
    data_A_F =      auto()
    data_time =     auto()

    # composition layers
    color_mask =    auto()
    flow =          auto()
    fluid =         auto()
    composite =     auto()


class LayerSettings(BaseSettings):
    """Which layers to draw in preview and final output (order matters)."""
    preview: Setting[list[Layers]] = Setting([Layers.composite], description="Layers drawn in the preview viewports")
    final:   Setting[list[Layers]] = Setting([Layers.composite], description="Layers drawn on the output monitors")
