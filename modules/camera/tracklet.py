"""A tracklet as a camera reports it: the device's own track of one person in one camera's frame."""

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from modules.utils import Rect

from .geometry import corrected_keypoints, corrected_roi, mirrored_keypoints, mirrored_roi


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
    to the frame; `timestamp` is the capture time of the frame the tracker looked at, the
    frame's own `CameraFrame.timestamp` (oak4, replay), else the message's arrival; the same for
    every tracklet of one message. `keypoints`
    is None unless the camera runs a pose model (OAK4): then float32 (17, 3), the body landmarks in
    the pose's order (`modules.pose.PointLandmark`, COCO), each x and y normalised to the frame
    like `roi`, and its score.
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
    message too. `straightened` says the boxes and keypoints are in the frame as the uploader makes
    it, straightened and mirrored; a camera hands them as its model found them, in the frame as
    delivered, and the uploader straightens them with their frame (`straightened_tracklets`).
    """
    cam_id: int
    timestamp: float
    tracklets: list[CameraTracklet]
    arrival: float | None = None
    straightened: bool = False


def straightened_tracklets(cam_tracklets: CameraTracklets, homography: np.ndarray | None, flip_h: bool) -> CameraTracklets:
    """The tracklets in the frame as the uploader makes it: straightened with their frame's
    homography, then mirrored when the frame is, as the image is."""
    tracklets: list[CameraTracklet] = []
    for tracklet in cam_tracklets.tracklets:
        roi, keypoints = tracklet.roi, tracklet.keypoints
        if homography is not None:
            roi, keypoints = corrected_roi(roi, homography), corrected_keypoints(keypoints, homography)
        if flip_h:
            roi, keypoints = mirrored_roi(roi), mirrored_keypoints(keypoints)
        tracklets.append(replace(tracklet, roi=roi, keypoints=keypoints))
    return replace(cam_tracklets, tracklets=tracklets, straightened=True)
