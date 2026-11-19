"""The values Superficial Flow plots over time, in lane order, and the timers that feed some of them."""

from enum import IntEnum, auto


class Series(IntEnum):
    CAMERA_INTERVAL =       0       # ms between video frames as the camera stamped them
    POSE_LATENCY_RAW =      auto()  # ms from a pose's capture to its keypoints out of the model (RAW stage)
    POSE_LATENCY_SMOOTH =   auto()  # ms from a pose's capture to the SMOOTH stage
    SEGMENTATION =          auto()  # ms the segmentation model takes on a frame's crops
    CAPTURE_LATENCY =       auto()  # ms from a capture (color and mask of one moment) to the board
    MASKS_UNSYNCED =        auto()  # 1 where a capture with a player carried an older mask than its own
    RENDER_INTERVAL =       auto()  # ms between render ticks
    WAIT_UPLOAD =           auto()  # ms a camera frame waits in the uploader's queue
    WAIT_UPLOAD_TRACKLETS = auto()  # ms a camera's tracklet message waits in the uploader's queue
    WAIT_CAPTURE_SYNC =     auto()  # ms an image, poses or masks wait in the capture sync's inbox
    WAIT_POSE_STEP =        auto()  # ms a frame bang waits for the pose step's thread
    WAIT_POSE_PIPELINE =    auto()  # ms a pose model's result waits for the stages after it


# PerformanceTimer names, as each timer is called, to the series their samples feed
TIMER_SERIES: dict[str, Series] = {
    'Render interval':          Series.RENDER_INTERVAL,
    'Segmentation':             Series.SEGMENTATION,
    'Wait upload':              Series.WAIT_UPLOAD,
    'Wait upload tracklets':    Series.WAIT_UPLOAD_TRACKLETS,
    'Wait inference frame sync': Series.WAIT_CAPTURE_SYNC,
    'Wait pose step':           Series.WAIT_POSE_STEP,
    'Wait pose pipeline':       Series.WAIT_POSE_PIPELINE,
}
