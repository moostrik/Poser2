"""One chunk's play: a reader for every camera's stream of the chunk, from one frame to another, its metadata and
its clock (`PlayClock`). The player holds the readers of the chunk playing, of the one loading and of those
closing; a chunk's readers load at once, play together and stop together."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from modules.video import StreamFrameCallback, StreamReader

from .chunk import ChunkInfo, ChunkMetadata, FrameMetadata, read_metadata
from .frame import FrameType
from .play_clock import PlayClock
from .recorder import make_chunk_name, make_file_name
from .recording_folder import RecordingFolder

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


class ChunkReaders:
    def __init__(self, folder: RecordingFolder, chunk: int, lo: int, hi: int | None, num_cams: int, types: list[FrameType],
                 frame_callback: StreamFrameCallback, decoding: 'Decoding', known: 'ChunkReaders | None' = None) -> None:
        """The readers of `folder`'s `chunk`, frames `lo` up to `hi` (None: to its end), of the streams of `types`
        it has, each handing its frames to `frame_callback` on its own thread. `known`: readers of the same chunk of
        the same folder, whose metadata is taken instead of read again (a loop over one chunk)."""
        self.folder: RecordingFolder = folder
        self.chunk: int = chunk
        self.types: set[FrameType] = set()          # the streams it plays: what a frame waits for
        self.metadata: dict[int, ChunkMetadata] = {}
        if known is not None and known.folder.path == folder.path and known.chunk == chunk:
            self.metadata = dict(known.metadata)
        else:
            for c in range(num_cams):
                path: Path = folder.path / make_chunk_name(c, chunk)
                if path.is_file():
                    self.metadata[c] = read_metadata(path, c)
        frames: dict[int, Sequence[FrameMetadata]] = {c: metadata.frames for c, metadata in self.metadata.items()}
        self.clock: PlayClock = PlayClock(frames, lo)
        self._readers: list[StreamReader] = []
        for c in range(num_cams):
            path = folder.path / make_chunk_name(c, chunk)
            if path.is_file():
                info: ChunkInfo | None = self.metadata[c].info if c in self.metadata else None
                with_depth: bool = info is None or info.depth
                for stream, t in enumerate((FrameType.VIDEO, FrameType.DEPTH)):
                    if t in types and (t == FrameType.VIDEO or with_depth):
                        self._load(c, t, path, stream, lo, hi, frame_callback, decoding, self.clock)
                continue
            for t in types:
                path = folder.path / make_file_name(c, t, chunk, decoding.suffix)
                if path.is_file():
                    self._load(c, t, path, 0, lo, hi, frame_callback, decoding, None)
                elif t == FrameType.VIDEO:
                    logger.warning(f"File {path} not found")

    def _load(self, cam_id: int, frame_type: FrameType, path: Path, stream: int, lo: int, hi: int | None,
              frame_callback: StreamFrameCallback, decoding: 'Decoding', clock: PlayClock | None) -> None:
        color: bool = frame_type == FrameType.VIDEO and not decoding.gray
        reader = StreamReader(cam_id, frame_type, cast(StreamFrameCallback, frame_callback), color, decoding.hwaccel,
                              decoding.hwaccel_device, decoding.fps, stream, frame_type == FrameType.DEPTH, clock,
                              color and decoding.nv12)
        reader.load(str(path), self.chunk, lo, hi)
        self._readers.append(reader)
        self.types.add(frame_type)

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

    def playing(self) -> bool:
        return any(reader.is_playing() for reader in self._readers)

    def stop(self) -> None:
        for reader in self._readers:
            reader.stop()

    def stopped(self) -> bool:
        return all(reader.is_stopped() for reader in self._readers)

    def join(self) -> None:
        for reader in self._readers:
            reader.join()


