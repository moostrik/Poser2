"""The best face of the last `window` seconds: a sliding window maximum over the frames' subjects.

A monotonic deque: a new entry drops every older entry it ranks at or above, so the front is the
best entry of the window and the deque holds only the entries that can still become the best,
a handful. Each entry keeps a reference to its camera frame, not a crop. Not thread-safe."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from modules.face.FaceDetector import Face


@dataclass(frozen=True)
class BestEntry:
    time: float
    accepted: bool      # passed every check
    quality: float
    image: np.ndarray   # the camera frame
    face: Face          # the subject in that frame, image pixels

    @property
    def rank(self) -> tuple[bool, float]:
        return (self.accepted, self.quality)


class BestFace:
    def __init__(self) -> None:
        self._entries: deque[BestEntry] = deque()

    def add(self, entry: BestEntry, window: float) -> None:
        """Enter a frame's subject; entries older than `window` before it are forgotten."""
        while self._entries and self._entries[-1].rank <= entry.rank:
            self._entries.pop()
        self._entries.append(entry)
        while self._entries[0].time < entry.time - window:
            self._entries.popleft()

    def best(self) -> BestEntry | None:
        return self._entries[0] if self._entries else None

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
