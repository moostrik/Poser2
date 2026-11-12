"""One chunk's play: a reader for every camera's stream of the chunk, from one moment to another, its metadata and its
clock (`PlayClock`). The player holds the readers of the chunk playing, of the one loading, of the one ending while
the next plays, and of those closing; a chunk's readers load at once and start together.

The range is given in camera 0's frames (the timeline's) and found in every other camera by recorded time: the
cameras of a rig run unsynced, so a camera's frame of the same round is up to half a frame period from camera 0's,
and each camera plays its frames captured from half a period before camera 0's first frame to half a period after
its last; a range from a chunk's start or to its end takes every camera's from its start or to its end, where the
recorder cut them all at one time. So cameras that dropped unlike frames start and end at one moment. A camera plays the streams it
has: color, and depth where its chunk holds depth."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from modules.video import StreamFrameCallback, StreamReader

from ..container.chunk import ChunkInfo, ChunkMetadata, FrameMetadata, make_chunk_name, make_file_name, read_metadata
from ...frame import CAPTURE_TOLERANCE, FrameType
from .play_clock import PlayClock
from ..container.recording_folder import RecordingFolder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decoding:
    """How a player's readers decode: to gray, to NV12 or to BGR, on which decoder, at which rate for a clip
    without times, and the color-only clips' suffix."""
    gray: bool
    nv12: bool
    hwaccel: str
    hwaccel_device: str
    fps: float
    suffix: str


def first_at(frames: Sequence[FrameMetadata], timestamp: float) -> int:
    """The first frame captured at or after `timestamp` (within the capture tolerance); `len(frames)` for none."""
    lo, hi = 0, len(frames)
    while lo < hi:
        mid: int = (lo + hi) // 2
        if frames[mid].timestamp < timestamp - CAPTURE_TOLERANCE:
            lo = mid + 1
        else:
            hi = mid
    return lo


class ChunkReaders:
    def __init__(self, folder: RecordingFolder, chunk: int, lo: int, hi: int | None, num_cams: int, types: list[FrameType],
                 frame_callback: StreamFrameCallback, decoding: Decoding, known: 'ChunkReaders | None' = None) -> None:
        """The readers of `folder`'s `chunk`, camera 0's frames `lo` up to `hi` (None: to its end) and every other
        camera's of the same moments, of the streams of `types` each camera has, each reader handing its frames to
        `frame_callback` on its own thread. `known`: readers of the same chunk of the same folder, whose metadata is
        taken instead of read again (a loop over one chunk, a seek in it)."""
        self.folder: RecordingFolder = folder
        self.chunk: int = chunk
        self.types: dict[int, set[FrameType]] = {}      # each camera's streams: what its frames wait for
        self.metadata: dict[int, ChunkMetadata] = {}
        if known is not None and known.folder.path == folder.path and known.chunk == chunk:
            self.metadata = dict(known.metadata)
        else:
            for c in range(num_cams):
                path: Path = folder.path / make_chunk_name(c, chunk)
                if path.is_file():
                    self.metadata[c] = read_metadata(path, c)
        ranges: dict[int, tuple[int, int | None]] = {c: self._range(c, lo, hi) for c in range(num_cams)}
        frames: dict[int, Sequence[FrameMetadata]] = {c: metadata.frames for c, metadata in self.metadata.items()}
        self.clock: PlayClock = PlayClock(frames, {c: first for c, (first, _) in ranges.items()})
        self._readers: list[StreamReader] = []
        for c in range(num_cams):
            first, end = ranges[c]
            path = folder.path / make_chunk_name(c, chunk)
            if path.is_file():
                info: ChunkInfo | None = self.metadata[c].info if c in self.metadata else None
                with_depth: bool = info is None or info.depth
                for stream, t in enumerate((FrameType.VIDEO, FrameType.DEPTH)):
                    if t in types and (t == FrameType.VIDEO or with_depth):
                        self._load(c, t, path, stream, first, end, frame_callback, decoding, self.clock)
                continue
            for t in types:
                path = folder.path / make_file_name(c, t, chunk, decoding.suffix)
                if path.is_file():
                    self._load(c, t, path, 0, first, end, frame_callback, decoding, None)
                elif t == FrameType.VIDEO:
                    logger.warning(f"File {path} not found")

    def _range(self, cam_id: int, lo: int, hi: int | None) -> tuple[int, int | None]:
        """Camera 0's frames `lo` to `hi` as this camera's, by recorded time; as they are without the times."""
        reference: ChunkMetadata | None = self.metadata.get(0)
        own: ChunkMetadata | None = self.metadata.get(cam_id)
        if cam_id == 0 or reference is None or own is None or not 0 <= lo < len(reference.frames):
            return lo, hi
        half: float = 0.5 / reference.info.fps if reference.info is not None and reference.info.fps > 0.0 else 0.0
        first: int = first_at(own.frames, reference.frames[lo].timestamp - half) if lo > 0 else 0
        if hi is None or not 0 < hi < len(reference.frames):
            return first, None
        end: int = first_at(own.frames, reference.frames[hi - 1].timestamp + half)
        return first, max(end, first)

    def _load(self, cam_id: int, frame_type: FrameType, path: Path, stream: int, lo: int, hi: int | None,
              frame_callback: StreamFrameCallback, decoding: Decoding, clock: PlayClock | None) -> None:
        color: bool = frame_type == FrameType.VIDEO and not decoding.gray
        reader = StreamReader(cam_id, frame_type, cast(StreamFrameCallback, frame_callback), color, decoding.hwaccel,
                              decoding.hwaccel_device, decoding.fps, stream, frame_type == FrameType.DEPTH, clock,
                              color and decoding.nv12)
        reader.load(str(path), self.chunk, lo, hi)
        self._readers.append(reader)
        self.types.setdefault(cam_id, set()).add(frame_type)

    def info(self, cam_id: int) -> ChunkInfo | None:
        """The chunk's fixed info for a camera; None for a recording without it."""
        metadata: ChunkMetadata | None = self.metadata.get(cam_id)
        return metadata.info if metadata is not None else None

    def frame(self, cam_id: int, frame_id: int) -> FrameMetadata | None:
        """What the chunk keeps of a camera's frame; None for a recording without it."""
        metadata: ChunkMetadata | None = self.metadata.get(cam_id)
        if metadata is None or not 0 <= frame_id < len(metadata.frames):
            return None
        return metadata.frames[frame_id]

    def loaded(self) -> bool:
        return all(reader.is_loaded() for reader in self._readers)

    def play(self) -> None:
        for reader in self._readers:
            reader.play()

    def finished(self) -> bool:
        """No reader plays any more."""
        return not any(reader.is_playing() for reader in self._readers)

    def stop(self) -> None:
        for reader in self._readers:
            reader.stop()

    def stopped(self) -> bool:
        return all(reader.is_stopped() for reader in self._readers)

    def join(self) -> None:
        for reader in self._readers:
            reader.join()
