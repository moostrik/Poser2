"""The hand-over of a play's frames: each camera's frame out once all its own streams are in (color and, where its
chunk has it, depth), with one stamp, its capture on the play's clock, and what the chunk keeps of it.

The streams' readers call `submit_frame` on their own threads. Each camera is handed over apart, under its own lock,
as live: one camera's frames never wait for another's. A camera's frames leave in their order, the chunk's and then
the frame's, so the chunk after the one playing can start beside it (`follow`) while the first ends; a frame older
than one gone out is dropped, as are frames left without their other stream, counted (`take_unpaired`). How far
the cameras are apart in recorded time is `drift`."""

import logging
from collections import defaultdict
from threading import Lock
from typing import Callable, TypeAlias

from numpy import ndarray

from .chunk import ChunkInfo, FrameMetadata
from .chunk_readers import ChunkReaders
from .frame import FrameType

logger = logging.getLogger(__name__)

# A clip frame as the player hands it over: (cam_id, frame_type, image, metadata, timestamp), the
# metadata the chunk keeps of the frame and its capture on the play's clock (`PlayClock.capture`).
# Both are None for a recording without metadata: the source it feeds stamps the frame at its
# hand-over then.
ClipFrameCallback: TypeAlias = Callable[[int, FrameType, ndarray, FrameMetadata | None, float | None], None]


class FrameHandOver:
    def __init__(self, num_cams: int, types: list[FrameType]) -> None:
        """`types`: the kinds in the order a frame's streams go out, VIDEO first: its readings set up what the
        depth of the frame uses."""
        self.types: list[FrameType] = types
        self.frame_callbacks: list[ClipFrameCallback] = []
        self._lock: Lock = Lock()                   # the chunks handed over and the counts
        self._camera_locks: dict[int, Lock] = defaultdict(Lock)     # one camera's frames out at a time, in order
        self._readers: dict[int, ChunkReaders] = {}         # the chunks whose frames go out, by number
        self._waiting: dict[int, dict[tuple[int, int], dict[FrameType, ndarray]]] = defaultdict(dict)
        self._handed_over: dict[int, tuple[int, int]] = {}  # each camera's last frame gone out: (chunk, frame)
        self._recorded: dict[int, float] = {}               # and its recorded capture time
        self._unpaired: int = 0                     # frames dropped without their other stream, since taken
        self.shown_frame: int = -1                  # camera 0's frame last gone out, for the panel

    def start(self, readers: ChunkReaders | None) -> None:
        """Hand over the frames of `readers`' chunk from here, and of no other; None hands over none."""
        with self._lock:
            self._readers = {readers.chunk: readers} if readers is not None else {}
            self._waiting.clear()
            self._handed_over.clear()
            self._recorded.clear()

    def follow(self, readers: ChunkReaders) -> None:
        """Hand over `readers`' chunk too, the one after the chunk playing: its frames go out after that chunk's."""
        with self._lock:
            self._readers[readers.chunk] = readers

    def retire(self, readers: ChunkReaders) -> None:
        """The chunk of `readers` ended: none of its frames go out any more."""
        with self._lock:
            if self._readers.get(readers.chunk) is readers:
                del self._readers[readers.chunk]

    def info(self, cam_id: int) -> ChunkInfo | None:
        """The fixed info of the chunk a camera's last frame came from; of the first chunk before any did."""
        with self._lock:
            handed: tuple[int, int] | None = self._handed_over.get(cam_id)
            readers: ChunkReaders | None = self._readers.get(handed[0]) if handed is not None else None
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

    def submit_frame(self, cam_id: int, frame_type: FrameType, frame: ndarray, chunk_id: int, frame_id: int) -> None:
        """A stream's frame, on its reader's thread (`StreamFrameCallback`)."""
        with self._lock:
            readers: ChunkReaders | None = self._readers.get(chunk_id)
        if readers is None:
            return
        if cam_id == 0 and frame_type == FrameType.VIDEO:
            self.shown_frame = frame_id
        key: tuple[int, int] = (chunk_id, frame_id)

        with self._camera_locks[cam_id]:
            with self._lock:
                cam_frames: dict[tuple[int, int], dict[FrameType, ndarray]] = self._waiting[cam_id]
                cam_frames.setdefault(key, {})[frame_type] = frame
                if any(t not in cam_frames[key] for t in readers.types.get(cam_id, ())):
                    return
                stale: list[tuple[int, int]] = [k for k in cam_frames if k < key]
                for k in stale:
                    del cam_frames[k]
                self._unpaired += len(stale)
                frames: dict[FrameType, ndarray] = cam_frames.pop(key)
                if key <= self._handed_over.get(cam_id, (-1, -1)):
                    return          # a frame older than one gone out: the order holds
                self._handed_over[cam_id] = key

            metadata: FrameMetadata | None = readers.frame(cam_id, frame_id)
            timestamp: float | None = readers.clock.capture(cam_id, frame_id)
            if metadata is not None:
                with self._lock:
                    self._recorded[cam_id] = metadata.timestamp
            for kind in self.types:
                if kind in frames:
                    for callback in self.frame_callbacks:
                        callback(cam_id, kind, frames[kind], metadata, timestamp)
