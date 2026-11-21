"""A tracklet as a camera reports it: the device's own track of one person in one camera's frame."""

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from modules.utils import Rect

from .geometry import corrected_keypoints, corrected_roi, flipped_keypoints, flipped_roi, mirrored_keypoints, mirrored_roi, \
    turned_keypoints, turned_roi


class TrackingStatus(Enum):
    """The state of a track, as the device's tracker reports it and as the host trackers carry it on."""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass(frozen=True)
class CameraTracklet:
    """What a camera's own tracker says about one player in one frame."""
    cam_id: int
    id: int                                 # the device tracker's id
    status: TrackingStatus
    roi: Rect                               # normalised to the frame
    age: int                                # frames
    timestamp: float                        # its frame's capture, as `CameraFrame.timestamp`
    keypoints: np.ndarray | None = None     # with a pose model on the camera: float32 (17, 3), COCO, x y normalised, score


@dataclass(frozen=True)
class CameraTracklets:
    """A camera's tracklets of one frame, as one message: what a tracklet callback gets."""
    cam_id: int
    timestamp: float                        # its frame's capture, as each tracklet's
    tracklets: list[CameraTracklet]         # may be empty: no player is a message too
    arrival: float | None = None            # its hand-over; None where `timestamp` is it
    straightened: bool = False              # in the frame as the uploader makes it, straightened, turned and flipped


def mirrored_tracklets(cam_tracklets: CameraTracklets) -> CameraTracklets:
    """The tracklets in their frame flipped left to right, not straightened: their boxes and keypoints mirrored."""
    tracklets: list[CameraTracklet] = [replace(tracklet, roi=mirrored_roi(tracklet.roi), keypoints=mirrored_keypoints(tracklet.keypoints))
                                       for tracklet in cam_tracklets.tracklets]
    return replace(cam_tracklets, tracklets=tracklets)


def straightened_tracklets(cam_tracklets: CameraTracklets, homography: np.ndarray | None, rotate90: bool, flip_v: bool,
                           flip_h: bool) -> CameraTracklets:
    """The tracklets in the frame as the uploader makes it, as the image is: straightened with their frame's
    homography, then turned and flipped as the frame is, in `CameraFrame`'s order."""
    tracklets: list[CameraTracklet] = []
    for tracklet in cam_tracklets.tracklets:
        roi, keypoints = tracklet.roi, tracklet.keypoints
        if homography is not None:
            roi, keypoints = corrected_roi(roi, homography), corrected_keypoints(keypoints, homography)
        if rotate90:
            roi, keypoints = turned_roi(roi), turned_keypoints(keypoints)
        if flip_v:
            roi, keypoints = flipped_roi(roi), flipped_keypoints(keypoints)
        if flip_h:
            roi, keypoints = mirrored_roi(roi), mirrored_keypoints(keypoints)
        tracklets.append(replace(tracklet, roi=roi, keypoints=keypoints))
    return replace(cam_tracklets, tracklets=tracklets, straightened=True)
