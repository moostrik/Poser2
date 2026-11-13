"""The join of the two-model chain's two streams per frame: the tracker's tracklets and the keypoints the pose
model found in each person's crop."""

from dataclasses import replace

import numpy as np

from .. import CameraTracklet

Box = tuple[float, float, float, float]      # xmin, ymin, xmax, ymax normalised to the delivered frame

BOX_MATCH_TOLERANCE: float = 1e-4            # the tracker's source detection is the detection, bit for bit


class KeypointJoin:
    """Puts each frame's keypoints on its tracklets, by the frame's sequence number and the tracklet's source box;
    a frame older than `keep` newer ones that never completed is dropped and counted (`dropped`)."""

    def __init__(self, keep: int = 8) -> None:
        self.keep: int = keep
        self.frames: dict[int, dict] = {}
        self.dropped: int = 0
        self.joined: int = 0

    def put_tracklets(self, seq: int, tracklets: list[CameraTracklet], boxes: list[Box]) -> list[CameraTracklet] | None:
        """The tracklets of frame `seq`, each with the box of its source detection; the joined tracklets when the
        frame's keypoints are in."""
        frame = self._frame(seq)
        frame['tracklets'] = (tracklets, boxes)
        return self._complete(seq)

    def put_keypoints(self, seq: int, boxes: list[Box], keypoints: list[np.ndarray | None]) -> list[CameraTracklet] | None:
        """The people of frame `seq`, each box with its keypoints (None when the crop gave none); the joined
        tracklets when the frame's tracklets are in."""
        frame = self._frame(seq)
        frame['keypoints'] = (boxes, keypoints)
        return self._complete(seq)

    def _frame(self, seq: int) -> dict:
        if seq not in self.frames:
            self.frames[seq] = {'tracklets': None, 'keypoints': None}
            for old in [s for s in self.frames if s < seq - self.keep]:
                del self.frames[old]
                self.dropped += 1
        return self.frames[seq]

    def _complete(self, seq: int) -> list[CameraTracklet] | None:
        frame = self.frames[seq]
        if frame['tracklets'] is None or frame['keypoints'] is None:
            return None
        del self.frames[seq]
        self.joined += 1
        tracklets, tracklet_boxes = frame['tracklets']
        boxes, keypoints = frame['keypoints']
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
