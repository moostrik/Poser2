"""The device tracker's statuses, as the contract names them: the `ObjectTracker` has exactly these four."""

import depthai as dai

from .. import TrackingStatus

TRACKING_STATUS: dict[dai.Tracklet.TrackingStatus, TrackingStatus] = {
    dai.Tracklet.TrackingStatus.NEW: TrackingStatus.NEW,
    dai.Tracklet.TrackingStatus.TRACKED: TrackingStatus.TRACKED,
    dai.Tracklet.TrackingStatus.LOST: TrackingStatus.LOST,
    dai.Tracklet.TrackingStatus.REMOVED: TrackingStatus.REMOVED,
}
