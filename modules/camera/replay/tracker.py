"""The replay backend's tracks: what its host tracker says about one person, in the shape the device
tracker reports (`CameraTracklet` without the camera and the time, which the camera adds)."""

from dataclasses import dataclass

from .. import TrackingStatus
from modules.utils import Rect


@dataclass(frozen=True)
class Track:
    id: int
    status: TrackingStatus
    roi: Rect
    age: int
