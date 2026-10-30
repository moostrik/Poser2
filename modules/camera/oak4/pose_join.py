"""The join of the two-model chain's two host streams: the tracker's tracklets and the gathered
poses of the crops, per frame.

Both reach the host for one frame with that frame's sequence number: the tracklets from the
device's object tracker, and the gathered poses, the people's boxes beside one keypoint set per
person in the same order (depthai-nodes' `GatherData`). A tracklet takes the keypoints of the person
whose box is its source detection. Plain data in, plain data out: no depthai here, so it is tested
without a device.
"""

from dataclasses import replace

import numpy as np

from .. import CameraTracklet

Box = tuple[float, float, float, float]      # xmin, ymin, xmax, ymax normalised to the delivered frame

BOX_MATCH_TOLERANCE: float = 1e-4            # the tracker's source detection is the detection, bit for bit


class PoseJoin:
    """Joins, per frame sequence number, the tracklets with the poses, into tracklets with keypoints.

    A frame is complete when both its streams are in; whichever arrives second gets the joined
    tracklets back. A frame older than `keep` newer ones that never completed is dropped and
    counted in `dropped`, so a stream that stalls does not pile up.
    """

    def __init__(self, keep: int = 8) -> None:
        self.keep: int = keep
        self.frames: dict[int, dict] = {}
        self.dropped: int = 0
        self.joined: int = 0

    def put_tracklets(self, seq: int, tracklets: list[CameraTracklet], boxes: list[Box]) -> list[CameraTracklet] | None:
        """The tracklets of frame `seq`, each with the box of its source detection."""
        frame = self._frame(seq)
        frame['tracklets'] = (tracklets, boxes)
        return self._complete(seq)

    def put_poses(self, seq: int, boxes: list[Box], keypoints: list[np.ndarray | None]) -> list[CameraTracklet] | None:
        """The people of frame `seq`, each box with its keypoints (None when the crop gave none)."""
        frame = self._frame(seq)
        frame['poses'] = (boxes, keypoints)
        return self._complete(seq)

    def _frame(self, seq: int) -> dict:
        if seq not in self.frames:
            self.frames[seq] = {'tracklets': None, 'poses': None}
            for old in [s for s in self.frames if s < seq - self.keep]:
                del self.frames[old]
                self.dropped += 1
        return self.frames[seq]

    def _complete(self, seq: int) -> list[CameraTracklet] | None:
        frame = self.frames[seq]
        if frame['tracklets'] is None or frame['poses'] is None:
            return None
        del self.frames[seq]
        self.joined += 1
        tracklets, tracklet_boxes = frame['tracklets']
        boxes, keypoints = frame['poses']
        joined: list[CameraTracklet] = []
        for tracklet, box in zip(tracklets, tracklet_boxes):
            index = match_box(box, boxes)
            points = keypoints[index] if index is not None else None
            joined.append(replace(tracklet, keypoints=points) if points is not None else tracklet)
        return joined


def match_box(box: Box, boxes: list[Box]) -> int | None:
    """The index of `box` among `boxes`, equal within `BOX_MATCH_TOLERANCE`; None when it is not there
    (a tracklet the tracker carries without a detection this frame: LOST, or REMOVED)."""
    for index, candidate in enumerate(boxes):
        if all(abs(a - b) <= BOX_MATCH_TOLERANCE for a, b in zip(box, candidate)):
            return index
    return None
