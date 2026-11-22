"""The hand-over of a play's frames: each camera's frame out once all its own streams are in (color and, where its
chunk has them, depth and its confidence), with one stamp, its capture on the play's clock, and what the chunk keeps
of it. The chunk's streams become the camera's frames here: the color a VIDEO frame, the depth a DEPTH frame with its
confidence, None where the chunk has none.

The streams' readers call `submit_frame` on their own threads. Each camera is handed over apart, under its own lock,
as live: one camera's frames never wait for another's. A camera's frames leave in their order, the chunk's and then
the frame's, so the chunk after the one playing can start beside it (`follow`) while the first ends; a frame older
than one gone out is dropped, as are frames left without their other stream, counted (`take_unpaired`). How far
the cameras are apart in recorded time is `drift`."""

import logging
from collections import defaultdict
from threading import Lock
from typing import Protocol

from numpy import ndarray

from modules.camera import ClipFrame, FrameType
from modules.utils import Broadcast

from ..chunk import ChunkInfo, ChunkStream, FrameMetadata
from .play_clock import PlayClock

logger = logging.getLogger(__name__)

FRAME_TYPES: dict[ChunkStream, FrameType] = {ChunkStream.VIDEO: FrameType.VIDEO, ChunkStream.DEPTH: FrameType.DEPTH}


class HandOverReaders(Protocol):
    """A chunk's readers as the hand-over sees them: the chunk's number, each camera's streams, a frame's metadata,
    the chunk's info and its clock (`ChunkReaders`)."""
    @property
    def chunk(self) -> int: ...
    @property
    def streams(self) -> dict[int, set[ChunkStream]]: ...
    @property
    def clock(self) -> PlayClock: ...
    def frame(self, cam_id: int, frame_id: int) -> FrameMetadata | None: ...
    def info(self, cam_id: int) -> ChunkInfo | None: ...


class FrameHandOver:
    def __init__(self, streams: list[ChunkStream]) -> None:
        """`streams`: in the order a frame's streams go out, VIDEO first: its readings set up what the depth of the
        frame uses."""
        self.streams: list[ChunkStream] = streams
        self.frame_callbacks: Broadcast[[ClipFrame]] = Broadcast(name='Player frame')
        self._lock: Lock = Lock()                   # the chunks handed over and the counts
        self._camera_locks: dict[int, Lock] = defaultdict(Lock)     # one camera's frames out at a time, in order
        self._readers: dict[int, HandOverReaders] = {}      # the chunks whose frames go out, by number
        self._waiting: dict[int, dict[tuple[int, int], dict[ChunkStream, ndarray]]] = defaultdict(dict)
        self._handed_over: dict[int, tuple[int, int]] = {}  # each camera's last frame gone out: (chunk, frame)
        self._recorded: dict[int, float] = {}               # and its recorded capture time
        self._unpaired: int = 0                     # frames dropped without their other stream, since taken
        self.shown_frame: int = -1                  # camera 0's frame last gone out, for the panel

    def start(self, readers: HandOverReaders | None) -> None:
        """Hand over the frames of `readers`' chunk from here, and of no other; None hands over none."""
        with self._lock:
            self._readers = {readers.chunk: readers} if readers is not None else {}
            self._waiting.clear()
            self._handed_over.clear()
            self._recorded.clear()

    def follow(self, readers: HandOverReaders) -> None:
        """Hand over `readers`' chunk too, the one after the chunk playing: its frames go out after that chunk's."""
        with self._lock:
            self._readers[readers.chunk] = readers

    def retire(self, readers: HandOverReaders) -> None:
        """The chunk of `readers` ended: none of its frames go out any more."""
        with self._lock:
            if self._readers.get(readers.chunk) is readers:
                del self._readers[readers.chunk]

    def info(self, cam_id: int) -> ChunkInfo | None:
        """The fixed info of the chunk a camera's last frame came from; of the first chunk before any did."""
        with self._lock:
            handed: tuple[int, int] | None = self._handed_over.get(cam_id)
            readers: HandOverReaders | None = self._readers.get(handed[0]) if handed is not None else None
            if readers is None and self._readers:
                readers = self._readers[min(self._readers)]
        return readers.info(cam_id) if readers is not None else None

    @property
    def drift(self) -> float:
        """How far apart the cameras' last frames gone out were captured, in ms."""
        with self._lock:
            times: list[float] = list(self._recorded.values())
        return (max(times) - min(times)) * 1000.0 if len(times) > 1 else 0.0

    def take_unpaired(self) -> int:
        """The frames dropped without their other stream since the last call."""
        with self._lock:
            unpaired, self._unpaired = self._unpaired, 0
        return unpaired

    def submit_frame(self, cam_id: int, stream: ChunkStream, frame: ndarray, chunk_id: int, frame_id: int) -> None:
        """A stream's frame, on its reader's thread (`StreamFrameCallback`)."""
        with self._lock:
            readers: HandOverReaders | None = self._readers.get(chunk_id)
        if readers is None:
            return
        if cam_id == 0 and stream == ChunkStream.VIDEO:
            self.shown_frame = frame_id
        key: tuple[int, int] = (chunk_id, frame_id)

        with self._camera_locks[cam_id]:
            with self._lock:
                cam_frames: dict[tuple[int, int], dict[ChunkStream, ndarray]] = self._waiting[cam_id]
                cam_frames.setdefault(key, {})[stream] = frame
                if any(s not in cam_frames[key] for s in readers.streams.get(cam_id, ())):
                    return
                stale: list[tuple[int, int]] = [k for k in cam_frames if k < key]
                for k in stale:
                    del cam_frames[k]
                self._unpaired += len(stale)
                frames: dict[ChunkStream, ndarray] = cam_frames.pop(key)
                if key <= self._handed_over.get(cam_id, (-1, -1)):
                    return          # a frame older than one gone out: the order holds
                self._handed_over[cam_id] = key

            metadata: FrameMetadata | None = readers.frame(cam_id, frame_id)
            timestamp: float | None = readers.clock.capture(cam_id, frame_id)
            if metadata is not None:
                with self._lock:
                    self._recorded[cam_id] = metadata.timestamp
            confidence: ndarray | None = frames.get(ChunkStream.CONFIDENCE)
            for stream in self.streams:
                kind: FrameType | None = FRAME_TYPES.get(stream)
                if kind is not None and stream in frames:
                    self.frame_callbacks(ClipFrame(cam_id, kind, frames[stream], confidence if kind == FrameType.DEPTH else None,
                                                   metadata, timestamp))
