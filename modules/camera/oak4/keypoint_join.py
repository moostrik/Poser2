"""The join of the two-model chain's two streams per frame: the tracker's tracklets and the keypoints the pose
model found in each player's crop."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock

import numpy as np

from .. import CameraTracklet

Box = tuple[float, float, float, float]      # xmin, ymin, xmax, ymax normalized to the delivered frame

BOX_MATCH_TOLERANCE: float = 1e-4            # the tracker's source detection is the detection, bit for bit


@dataclass
class _Halves:
    """One frame's two streams as they come in: its tracklets with their source boxes, its boxes with their
    keypoints; None until that stream's message is in."""
    tracklets: tuple[list[CameraTracklet], list[Box]] | None = None
    keypoints: tuple[list[Box], list[np.ndarray | None]] | None = None


class KeypointJoin:
    """Puts each frame's keypoints on its tracklets, by the frame's sequence number and the tracklet's source box;
    a frame older than `keep` newer ones that never completed is dropped and counted once (`dropped`), and a half
    that comes after its frame was dropped is turned away. The two streams come from two output queues, whose
    callbacks run on two threads: the frames and the count are kept under a lock, and `on_dropped` gets each new
    count under it, so a newer count is never written over by an older one."""

    def __init__(self, keep: int = 8, on_dropped: Callable[[int], None] | None = None) -> None:
        self.keep: int = keep
        self.on_dropped: Callable[[int], None] | None = on_dropped
        self._lock: Lock = Lock()
        self.frames: dict[int, _Halves] = {}
        self.dropped: int = 0
        self.joined: int = 0
        self._newest_dropped: int = -1      # the newest frame number dropped; a half at or below it comes too late

    def put_tracklets(self, seq: int, tracklets: list[CameraTracklet], boxes: list[Box]) -> list[CameraTracklet] | None:
        """The tracklets of frame `seq`, each with the box of its source detection; the joined tracklets when the
        frame's keypoints are in."""
        with self._lock:
            halves: _Halves | None = self._frame(seq)
            if halves is None:
                return None
            halves.tracklets = (tracklets, boxes)
            return self._complete(seq, halves)

    def put_keypoints(self, seq: int, boxes: list[Box], keypoints: list[np.ndarray | None]) -> list[CameraTracklet] | None:
        """The players of frame `seq`, each box with its keypoints (None when the crop gave none); the joined
        tracklets when the frame's tracklets are in."""
        with self._lock:
            halves: _Halves | None = self._frame(seq)
            if halves is None:
                return None
            halves.keypoints = (boxes, keypoints)
            return self._complete(seq, halves)

    def clear(self) -> None:
        """Empties the join for a new pipeline, whose frame numbers may start again: each half-frame left counts as
        dropped, and no number is too late any more."""
        with self._lock:
            left: int = len(self.frames)
            self.frames.clear()
            self._newest_dropped = -1
            if left:
                self._count_dropped(left)

    def _frame(self, seq: int) -> _Halves | None:
        """Frame `seq`'s halves; None when the frame was dropped already."""
        if seq <= self._newest_dropped:
            return None
        if seq not in self.frames:
            self.frames[seq] = _Halves()
            old: list[int] = [s for s in self.frames if s < seq - self.keep]
            for s in old:
                del self.frames[s]
            if old:
                self._newest_dropped = max(old)
                self._count_dropped(len(old))
        return self.frames[seq]

    def _count_dropped(self, count: int) -> None:
        self.dropped += count
        if self.on_dropped is not None:
            self.on_dropped(self.dropped)

    def _complete(self, seq: int, halves: _Halves) -> list[CameraTracklet] | None:
        if halves.tracklets is None or halves.keypoints is None:
            return None
        del self.frames[seq]
        self.joined += 1
        tracklets, tracklet_boxes = halves.tracklets
        boxes, keypoints = halves.keypoints
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
