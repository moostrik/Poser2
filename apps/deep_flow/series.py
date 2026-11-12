"""The values Deep Flow plots over time, in lane order, and the timers that feed some of them."""

from enum import IntEnum, auto


class Series(IntEnum):
    CAMERA_INTERVAL =       0       # ms between video frames as the camera stamped them
    POSE_LATENCY_RAW =      auto()  # ms from a pose's capture to its keypoints out of the model (RAW stage)
    POSE_LATENCY_SMOOTH =   auto()  # ms from a pose's capture to the SMOOTH stage
    CAPTURE_LATENCY =       auto()  # ms from a capture (color, depth, poses of one moment) to the board
    POSES_UNSYNCED =        auto()  # 1 where a capture with a player carried older poses than its own
    RENDER_INTERVAL =       auto()  # ms between render ticks
    GPU_BRIDGE =            auto()  # ms of GPU in a flow layer's bridges, A and B alike
    GPU_FLUID =             auto()  # ms of GPU in a fluid's step


# PerformanceTimer names, as each timer is called, to the series their samples feed
TIMER_SERIES: dict[str, Series] = {
    'Render interval':  Series.RENDER_INTERVAL,
    'GPU bridge':       Series.GPU_BRIDGE,
    'GPU fluid':        Series.GPU_FLUID,
}
