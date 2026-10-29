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
    to the delivered frame; `time` is the host arrival time of the message, stamped by the backend
    at hand-over. `keypoints` is empty until a backend fills it (OAK4).
    """
    cam_id: int
    id: int
    status: TrackingStatus
    roi: Rect
    age: int
    time: float
    keypoints: np.ndarray | None = None
