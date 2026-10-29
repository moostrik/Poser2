"""The replay backend's tracker: the device's object tracker in software, in SORT's shape.

The device (`ZERO_TERM_IMAGELESS`) matches each frame's detections to its tracks by box position
and shape with a Hungarian assignment and smooths them with a Kalman filter; so does this. A
track is NEW the frame it is first seen, TRACKED while a detection matches it, LOST while none
does (its last box stands, nothing is extrapolated: zero-term), and REMOVED, reported once, after
`lost_frames` without a match. Ids are the smallest free ones, as the pipeline builder sets on
the device (`SMALLEST_ID`); `age` counts the frames since the track was born. Boxes are fractions
of the frame throughout.
"""

from dataclasses import dataclass

import numpy as np

from modules.utils import Rect

from .. import TrackingStatus

LOST_FRAMES: int = 60       # two seconds at 30 fps; the device's count is in its firmware (v3 documents 120)
MIN_OVERLAP: float = 0.3    # a detection and a track overlapping less (IoU) are never matched, SORT's value
_SCALE: float = 1000.0      # the filter runs in a 1000 x 1000 frame, where SORT's noise figures hold


@dataclass(frozen=True)
class Track:
    id: int
    status: TrackingStatus
    roi: Rect
    age: int


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of every box of `a` (N, 4) with every box of `b` (M, 4), boxes as
    x0, y0, x1, y1: (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32)


def assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """The Hungarian assignment of rows to columns of `cost` (N, M) with the least total cost: the
    (row, column) pairs, min(N, M) of them. The O(n^3) shortest augmenting path form."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    transposed: bool = n > m
    if transposed:
        cost = cost.T
        n, m = m, n
    INF = float('inf')
    u = [0.0] * (n + 1)             # potentials
    v = [0.0] * (m + 1)
    way = [0] * (m + 1)
    match = [0] * (m + 1)           # match[column] = row (1-based), 0 = none
    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = match[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = float(cost[i0 - 1, j - 1]) - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
    pairs: list[tuple[int, int]] = [(match[j] - 1, j - 1) for j in range(1, m + 1) if match[j] > 0]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    return sorted(pairs)


class _Kalman:
    """SORT's constant-velocity filter on a box as centre, area and aspect ratio."""

    def __init__(self, box: np.ndarray) -> None:
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0            # the velocities start unknown
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros(7)
        self.x[:4] = _to_state(box)

    def predict(self) -> np.ndarray:
        if self.x[6] + self.x[2] <= 0:      # the area would go negative: hold it
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _to_box(self.x[:4])

    def update(self, box: np.ndarray) -> None:
        z = _to_state(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


def _to_state(box: np.ndarray) -> np.ndarray:
    """x0, y0, x1, y1 to centre x, centre y, area, aspect ratio."""
    w = box[2] - box[0]
    h = box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / h if h > 0 else 0.0])


def _to_box(state: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(state[2] * state[3], 0.0))
    h = state[2] / w if w > 0 else 0.0
    return np.array([state[0] - w / 2, state[1] - h / 2, state[0] + w / 2, state[1] + h / 2])


class _Tracked:
    def __init__(self, id: int, box: np.ndarray) -> None:
        self.id: int = id
        self.filter: _Kalman = _Kalman(box)
        self.box: np.ndarray = box          # the last measured box
        self.age: int = 1                   # the birth frame counts
        self.lost: int = 0                  # frames since the last match
        self.status: TrackingStatus = TrackingStatus.NEW


class Tracker:
    def __init__(self, lost_frames: int = LOST_FRAMES, min_overlap: float = MIN_OVERLAP) -> None:
        self.lost_frames: int = lost_frames
        self.min_overlap: float = min_overlap
        self._tracks: list[_Tracked] = []

    def update(self, boxes: list[Rect]) -> list[Track]:
        """The frame's detections in, the tracks out, in id order; a removed track is in the list
        once more, as REMOVED, and gone the frame after."""
        detections: np.ndarray = np.array([[r.x, r.y, r.x + r.width, r.y + r.height] for r in boxes], dtype=np.float64).reshape(-1, 4) * _SCALE
        self._tracks = [t for t in self._tracks if t.status != TrackingStatus.REMOVED]
        predicted: np.ndarray = np.array([t.filter.predict() for t in self._tracks]).reshape(-1, 4)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        if len(self._tracks) and len(detections):
            iou: np.ndarray = overlap(predicted, detections)
            for row, column in assign(1.0 - iou):
                if iou[row, column] >= self.min_overlap:
                    matched_tracks.add(row)
                    matched_detections.add(column)
                    track = self._tracks[row]
                    track.filter.update(detections[column])
                    track.box = detections[column]
                    track.lost = 0
                    track.status = TrackingStatus.TRACKED
        for row, track in enumerate(self._tracks):
            track.age += 1
            if row in matched_tracks:
                continue
            track.lost += 1
            track.status = TrackingStatus.REMOVED if track.lost > self.lost_frames else TrackingStatus.LOST
        for column in range(len(detections)):
            if column not in matched_detections:
                self._tracks.append(_Tracked(self._free_id(), detections[column]))
        self._tracks.sort(key=lambda t: t.id)
        return [Track(id=t.id, status=t.status, roi=_to_rect(t.box), age=t.age) for t in self._tracks]

    def _free_id(self) -> int:
        taken: set[int] = {t.id for t in self._tracks if t.status != TrackingStatus.REMOVED}
        id: int = 0
        while id in taken:
            id += 1
        return id


def _to_rect(box: np.ndarray) -> Rect:
    x0, y0, x1, y1 = (float(v) / _SCALE for v in box)
    return Rect(x0, y0, x1 - x0, y1 - y0)
