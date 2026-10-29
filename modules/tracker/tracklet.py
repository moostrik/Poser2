# Standard library imports
from dataclasses import dataclass, field
from typing import Callable, TypeAlias

# Local application imports
from modules.camera import CameraTracklet, TrackingStatus
from modules.utils import Rect
from .tracker_base import TrackerAnnotation


@dataclass(frozen=True)
class Tracklet:
    """A person as a host tracker knows them: the world identity it decides on top of what the
    camera reports (`CameraTracklet`), with its own timing and annotation. Its times are on the
    one clock: `from_camera` takes the message's host time, and expiry is judged against the time
    of the tick that asks (`is_expired`), never a clock read."""
    cam_id: int
    id: int = field(default=-1)          # world identity in a tracker's store; the player slot on its emitted copies
    obs_id: int = field(default=-1)      # this observation's identity, assigned by a store

    time_stamp: float = field(kw_only=True)     # the message this observation came with
    created_at: float = field(kw_only=True)
    last_active: float = field(kw_only=True)    # Last time person was actually detected (NEW/TRACKED)

    status: TrackingStatus = field(default=TrackingStatus.NEW)
    roi: Rect = field(default_factory=Rect)

    annotation: TrackerAnnotation | None = field(default=None)
    external_id: int = field(default=-1)
    external_age_in_frames: int = field(default=0)

    @property
    def is_lost(self) -> bool:
        return self.status == TrackingStatus.LOST

    @property
    def is_removed(self) -> bool:
        return self.status == TrackingStatus.REMOVED

    @property
    def is_active(self) -> bool:
        return self.status in (TrackingStatus.NEW, TrackingStatus.TRACKED)

    @property
    def age_in_seconds(self) -> float:
        """Get how long this person has been tracked"""
        return self.last_active - self.created_at

    def is_expired(self, now: float, threshold: float) -> bool:
        """True when the tracklet has not been updated within `threshold` seconds of `now`, the
        time of the frame or tick that asks."""
        return (now - self.last_active) > threshold

    @classmethod
    def from_camera(cls, tracklet: CameraTracklet) -> 'Tracklet':
        """The tracklet a camera reports, as the tracker takes it in."""
        return cls(
            cam_id=tracklet.cam_id,
            time_stamp=tracklet.time,
            created_at=tracklet.time,
            last_active=tracklet.time,
            status=tracklet.status,
            roi=tracklet.roi,
            external_id=tracklet.id,
            external_age_in_frames=tracklet.age,
        )



# Type Aliases
TrackletCallback: TypeAlias = Callable[[Tracklet], None]
TrackletDict: TypeAlias = dict[int, Tracklet]
TrackletDictCallback: TypeAlias = Callable[[TrackletDict], None]
TrackletListCallback: TypeAlias = Callable[[list[Tracklet]], None]
