"""The hand-over of a chunk's frames: each camera's frame out once all its streams are in (color and, where the chunk
has it, depth), with one stamp, its capture on the play's clock, and what the chunk keeps of it.

The streams' readers call `submit_frame` on their own threads; a complete frame goes out under one lock, so frames
leave in their order, and one older than a frame gone out is dropped, as are frames left without their other
stream, counted (`take_unpaired`). How far the cameras are apart in frames is `drift`."""

import logging
from threading import Lock
from typing import Callable, TypeAlias

from numpy import ndarray

from .chunk import FrameMetadata
from .chunk_readers import ChunkReaders
from .frame import FrameType

logger = logging.getLogger(__name__)

# A clip frame as the player hands it over: (cam_id, frame_type, image, metadata, timestamp), the
# metadata the chunk keeps of the frame and its capture on the play's clock (`PlayClock.capture`).
# Both are None for a recording without metadata: the source it feeds stamps the frame at its
# hand-over then.
ClipFrameCallback: TypeAlias = Callable[[int, FrameType, ndarray, FrameMetadata | None, float | None], None]


class FramePairing:
    def __init__(self, num_cams: int, types: list[FrameType]) -> None:
        """`types`: the kinds in the order a frame's streams go out, VIDEO first: its readings set up what the
        depth of the frame uses."""
        self.types: list[FrameType] = types
        self.frame_callbacks: list[ClipFrameCallback] = []
        self._lock: Lock = Lock()                   # the frames waiting and the counts
        self._hand_over_lock: Lock = Lock()         # one frame's streams out at a time, in frame order
        self._readers: ChunkReaders | None = None   # the chunk playing
        self._waiting: dict[int, dict[int, dict[FrameType, ndarray]]] = {c: {} for c in range(num_cams)}
        self._handed_over: dict[int, int] = {}      # each camera's last frame gone out
        self._unpaired: int = 0                     # frames dropped without their other stream, since taken
        self._drift_ids: dict[int, int] = {}
        self._drift: int = 0
        self.shown_frame: int = -1                  # camera 0's frame last gone out, for the panel

    def start(self, readers: ChunkReaders | None) -> None:
        """Hand over the frames of `readers`' chunk from here; None hands over none."""
        with self._lock:
            self._readers = readers
            for frames in self._waiting.values():
                frames.clear()
            self._handed_over.clear()
            self._drift_ids.clear()
            self._drift = 0

    @property
    def chunk(self) -> int:
        """The chunk whose frames go out; -1 for none."""
        with self._lock:
            return self._readers.chunk if self._readers is not None else -1

    @property
    def drift(self) -> int:
        """How many frames apart the cameras' last frames gone out are."""
        with self._lock:
            return self._drift

    def take_unpaired(self) -> int:
        """The frames dropped without their other stream since the last call."""
        with self._lock:
            unpaired, self._unpaired = self._unpaired, 0
        return unpaired

    def submit_frame(self, cam_id: int, frame_type: FrameType, frame: ndarray, chunk_id: int, frame_id: int) -> None:
        """A stream's frame, on its reader's thread (`StreamFrameCallback`)."""
        with self._lock:
            readers: ChunkReaders | None = self._readers
        if readers is None or chunk_id != readers.chunk:
            return
        if cam_id == 0 and frame_type == FrameType.VIDEO:
            self.shown_frame = frame_id

        with self._hand_over_lock:
            with self._lock:
                cam_frames: dict[int, dict[FrameType, ndarray]] = self._waiting.setdefault(cam_id, {})
                cam_frames.setdefault(frame_id, {})[frame_type] = frame
                if any(t not in cam_frames[frame_id] for t in readers.types):
                    return
                stale: list[int] = [key for key in cam_frames if key < frame_id]
                for key in stale:
                    del cam_frames[key]
                self._unpaired += len(stale)
                frames: dict[FrameType, ndarray] = cam_frames.pop(frame_id)
                if frame_id <= self._handed_over.get(cam_id, -1):
                    return          # a frame older than one gone out: the order holds
                self._handed_over[cam_id] = frame_id
                self._drift_ids[cam_id] = frame_id
                self._drift = max(self._drift_ids.values()) - min(self._drift_ids.values())

            metadata: FrameMetadata | None = readers.frame(cam_id, frame_id)
            timestamp: float | None = readers.clock.capture(cam_id, frame_id)
            for kind in self.types:
                if kind in frames:
                    for callback in self.frame_callbacks:
                        callback(cam_id, kind, frames[kind], metadata, timestamp)
