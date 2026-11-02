"""The values Deep Flow plots over time, in lane order, and the timers that feed some of them."""

from enum import IntEnum, auto


class Series(IntEnum):
    CAMERA_INTERVAL =       0       # ms between video frames as the camera stamped them
    SEGMENTATION_INTERVAL = auto()  # ms between results that carry masks
    SEGMENTATION_LATENCY =  auto()  # ms from the camera frame to its masks on the board
    SEGMENTATION_TIME =     auto()  # ms of inference per batch
    MASK_MISSING =          auto()  # 1 where a result carried no mask while there were players
    RENDER_INTERVAL =       auto()  # ms between render ticks
    GPU_BRIDGE =            auto()  # ms of GPU in a flow layer's bridges, A and B alike
    GPU_FLUID =             auto()  # ms of GPU in a fluid's step


# PerformanceTimer names, as each timer is called, to the series their samples feed
TIMER_SERIES: dict[str, Series] = {
    'RVM Segmentation': Series.SEGMENTATION_TIME,
    'Render interval':  Series.RENDER_INTERVAL,
    'GPU bridge':       Series.GPU_BRIDGE,
    'GPU fluid':        Series.GPU_FLUID,
}
