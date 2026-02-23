"""Layer identifiers for the render pipeline."""

from enum import IntEnum, auto


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
    centre_math=    auto()
    centre_cam =    auto()
    centre_mask =   auto()
    centre_frg =    auto()
    centre_pose =   auto()

    # Data layers (configurable slots A and B, all pre-allocated)
    data_B_W =      auto()
    data_B_F =      auto()
    data_A_W =      auto()
    data_A_F =      auto()
    data_time =     auto()

    # composition layers
    motion =        auto()
    flow =          auto()
    fluid =         auto()
    ms_mask =       auto()
    composite =     auto()
