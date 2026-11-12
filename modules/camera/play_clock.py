"""The recorded time of a play: when the player hands each recorded frame over, on `perf_counter`."""

from collections.abc import Sequence
from threading import Lock

from .chunk import FrameMetadata


class PlayClock:
    """The recorded time of one chunk's play, on `perf_counter`: the `FrameSchedule` every reader of the chunk
    shares. From `start` frame `n` of a camera is captured at the origin plus its recorded capture time minus the
    base, the first frame's (`capture`), and due at that plus its recorded latency (`moment`), so a frame comes as
    late after its capture as it came live, and a frame lost live leaves its gap. A reader that falls behind moves
    the origin for every reader, as far as its frame is still behind (`catch_up`). The chunk after it in the
    recording carries its origin and base on (`carry_on`), so the recorded time runs on across the two. Frames the
    chunk has no time for are not on it: their readers pace at `fps`."""

    def __init__(self, frames: dict[int, Sequence[FrameMetadata]], first: int) -> None:
        """`frames`: each camera's metadata by frame number; `first`: the frame number the play starts at."""
        self._frames: dict[int, Sequence[FrameMetadata]] = frames
        starts: list[float] = [cam_frames[first].timestamp for cam_frames in frames.values() if first < len(cam_frames)]
        self._base: float | None = min(starts) if starts else None
        self._origin: float | None = None
        self._lock: Lock = Lock()

    def start(self, origin: float) -> None:
        with self._lock:
            self._origin = origin

    def carry_on(self, before: 'PlayClock', now: float) -> None:
        """Run on from `before`, the clock of the chunk played before this one in the recording: its origin and
        base, when it has both and this chunk has times; else a new start at `now`."""
        with before._lock:
            origin, base = before._origin, before._base
        with self._lock:
            if origin is None or base is None or self._base is None:
                self._origin = now
                return
            self._origin, self._base = origin, base

    def catch_up(self, cam_id: int, frame_id: int, now: float) -> float:
        """The frame found overdue at `now`: the origin moves as far as the frame is still behind, and its moment
        after it."""
        with self._lock:
            due: float | None = self._moment(cam_id, frame_id)
            if due is None or self._origin is None:
                return now
            late: float = now - due
            if late > 0.0:
                self._origin += late
            return due + max(late, 0.0)

    def capture(self, cam_id: int, frame_id: int) -> float | None:
        """When the frame was captured, on the play's clock."""
        with self._lock:
            return self._capture(cam_id, frame_id)

    def moment(self, cam_id: int, frame_id: int) -> float | None:
        """When the frame is handed over: its capture plus the latency it came with live."""
        with self._lock:
            return self._moment(cam_id, frame_id)

    def _capture(self, cam_id: int, frame_id: int) -> float | None:
        frame: FrameMetadata | None = self._frame(cam_id, frame_id)
        if frame is None or self._origin is None or self._base is None:
            return None
        return self._origin + frame.timestamp - self._base

    def _moment(self, cam_id: int, frame_id: int) -> float | None:
        captured: float | None = self._capture(cam_id, frame_id)
        if captured is None:
            return None
        frame: FrameMetadata | None = self._frame(cam_id, frame_id)
        latency: float | None = frame.video_latency if frame is not None else None
        return captured + (latency or 0.0)

    def _frame(self, cam_id: int, frame_id: int) -> FrameMetadata | None:
        cam_frames: Sequence[FrameMetadata] | None = self._frames.get(cam_id)
        if cam_frames is None or not 0 <= frame_id < len(cam_frames):
            return None
        return cam_frames[frame_id]
