"""The recorded time of a play: when the player hands each recorded frame over, on `perf_counter`."""

from collections.abc import Mapping, Sequence
from threading import Lock

from ..chunk import FrameMetadata


class _Origin:
    """A play's origin on `perf_counter`, shared by the clocks of the chunks that run on from each other, so a
    catch-up by one reader moves them all."""

    def __init__(self, origin: float) -> None:
        self.lock: Lock = Lock()
        self.origin: float = origin


class PlayClock:
    """The recorded time of one chunk's play, on `perf_counter`: the `FrameSchedule` every reader of the chunk
    shares. From `start` frame `n` of a camera is captured at the origin plus its recorded capture time minus the
    base, the earliest first frame's of the cameras (`capture`), and due at that plus its recorded latency
    (`moment`), so a frame comes as late after its capture as it came live, and a frame lost live leaves its gap. A
    reader that falls behind moves the origin for every reader, as far as its frame is still behind (`catch_up`).
    The chunk after it in the recording carries its origin and base on (`carry_on`): the two clocks share the
    origin, so the recorded time runs on across the two while both play. A camera without recorded times (an old
    clip) is paced on the same origin at `fps`: frame `n` captured, and due, at the origin plus `n` from its first
    over `fps`, so the cameras of such a recording come as a synced rig's do."""

    def __init__(self, frames: Mapping[int, Sequence[FrameMetadata]], firsts: dict[int, int], fps: float = 0.0) -> None:
        """`frames`: each camera's metadata by frame number; `firsts`: each camera's frame number the play starts
        at; `fps`: the pace of a camera without recorded times, 0 for none."""
        self._frames: Mapping[int, Sequence[FrameMetadata]] = frames
        self._firsts: dict[int, int] = firsts
        self._fps: float = fps
        starts: list[float] = [cam_frames[firsts.get(c, 0)].timestamp for c, cam_frames in frames.items()
                               if 0 <= firsts.get(c, 0) < len(cam_frames)]
        self._base: float | None = min(starts) if starts else None
        self._origin: _Origin | None = None

    @property
    def has_times(self) -> bool:
        """Whether the chunk has recorded times: only then can the chunk after it run on from them."""
        return self._base is not None

    def start(self, origin: float) -> None:
        self._origin = _Origin(origin)

    def carry_on(self, before: 'PlayClock', now: float) -> None:
        """Run on from `before`, the clock of the chunk played before this one in the recording: its origin, shared,
        and its base, when it has both and this chunk has times; else a new start at `now`."""
        if before._origin is None or before._base is None or self._base is None:
            self.start(now)
            return
        self._origin, self._base = before._origin, before._base

    def catch_up(self, cam_id: int, frame_id: int, now: float) -> float:
        """The frame found overdue at `now`: the origin moves as far as the frame is still behind, and its moment
        after it."""
        origin: _Origin | None = self._origin
        if origin is None:
            return now
        with origin.lock:
            due: float | None = self._moment(cam_id, frame_id, origin.origin)
            if due is None:
                return now
            late: float = now - due
            if late > 0.0:
                origin.origin += late
            return due + max(late, 0.0)

    def capture(self, cam_id: int, frame_id: int) -> float | None:
        """When the frame was captured, on the play's clock."""
        origin: _Origin | None = self._origin
        if origin is None:
            return None
        with origin.lock:
            return self._capture(cam_id, frame_id, origin.origin)

    def moment(self, cam_id: int, frame_id: int) -> float | None:
        """When the frame is handed over: its capture plus the latency it came with live."""
        origin: _Origin | None = self._origin
        if origin is None:
            return None
        with origin.lock:
            return self._moment(cam_id, frame_id, origin.origin)

    def _capture(self, cam_id: int, frame_id: int, origin: float) -> float | None:
        if cam_id not in self._frames:
            return origin + (frame_id - self._firsts.get(cam_id, 0)) / self._fps if self._fps > 0.0 else None
        frame: FrameMetadata | None = self._frame(cam_id, frame_id)
        if frame is None or self._base is None:
            return None
        return origin + frame.timestamp - self._base

    def _moment(self, cam_id: int, frame_id: int, origin: float) -> float | None:
        captured: float | None = self._capture(cam_id, frame_id, origin)
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
