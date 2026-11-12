"""A tracklet as a camera reports it: the device's own track of one person in one camera's frame."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from modules.utils import Rect


class TrackingStatus(Enum):
    """The state of a track, as the device's tracker reports it and as the host trackers carry it on."""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass(frozen=True)
class CameraTracklet:
    """Device-neutral: what a camera's object tracker says about one track, in its own frame.

    `id` is the device tracker's id for the track and `age` its age in frames; `roi` is normalised
    to the delivered frame; `timestamp` is the capture time of the frame the tracker looked at, the
    frame's own `CameraFrame.timestamp` (oak4, replay), else the message's arrival; the same for
    every tracklet of one message. `keypoints`
    is None unless the camera runs a pose model (OAK4): then float32 (17, 3), the body landmarks in
    the pose's order (`modules.pose.PointLandmark`, COCO), each x and y normalised to the delivered
    frame like `roi`, and its score.
    """
    cam_id: int
    id: int
    status: TrackingStatus
    roi: Rect
    age: int
    timestamp: float
    keypoints: np.ndarray | None = None


@dataclass(frozen=True)
class CameraTracklets:
    """A camera's tracklets of one frame, as one message: what a tracklet callback gets.

    `timestamp` is the capture time of the frame the tracker looked at, the same as each
    tracklet's; `arrival` the moment the backend handed the message over, on the same clock, None
    where the backend stamps the arrival as the capture. `tracklets` may be empty: no player is a
    message too.
    """
    cam_id: int
    timestamp: float
    tracklets: list[CameraTracklet]
    arrival: float | None = None
