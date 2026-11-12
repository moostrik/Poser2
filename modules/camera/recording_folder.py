"""A folder of recordings as the player sees it: the folders found, a folder's chunks and their frame counts, its in
and out markers in a file of its own, and where a place on its timeline falls, as a chunk and a frame in it.

The markers belong to the recording: each folder keeps its own in `IN_OUT_FILE`, so they survive a restart and a
change of preset. The timeline (`Timeline`) is plain arithmetic on the chunks' frame counts: no thread, no reader."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from modules.video import CoderFormat, StreamReader

from .frame import FrameType
from .recorder import make_chunk_name, make_file_name

logger = logging.getLogger(__name__)

IN_OUT_FILE: str = 'in_out.json'        # in a recording's folder: its markers


def read_in_out(folder: Path) -> list[float]:
    """A recording's markers; all of it when it has none, or they cannot be read."""
    path: Path = folder / IN_OUT_FILE
    if not path.is_file():
        return [0.0, 1.0]
    try:
        markers: dict = json.loads(path.read_text(encoding='utf-8'))
        return [float(markers['in']), float(markers['out'])]
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("Could not read the markers in %s", path)
        return [0.0, 1.0]


def write_in_out(folder: Path, in_out: list[float]) -> None:
    try:
        (folder / IN_OUT_FILE).write_text(json.dumps({'in': in_out[0], 'out': in_out[1]}), encoding='utf-8')
    except OSError:
        logger.exception("Could not save the markers in %s", folder)


@dataclass(frozen=True)
class RecordingFolder:
    """One recording: its folder and its last chunk's number."""
    name: str
    path: Path
    chunks: int

    @staticmethod
    def find(video_path: Path) -> dict[str, 'RecordingFolder']:
        """The recordings in `video_path`: every folder with a chunk, `0_000.mkv` or a color-only `0_C_000.mp4`."""
        folders: dict[str, RecordingFolder] = {}
        if not video_path.is_dir():
            return folders
        for folder in video_path.iterdir():
            if folder.is_dir():
                max_chunk: int = -1
                for file in folder.iterdir():
                    if file.is_file() and file.suffix in ('.mkv', CoderFormat.H264.value, CoderFormat.H265.value):
                        n: str = file.stem.split('_')[-1]
                        if n.isdigit():
                            max_chunk = max(max_chunk, int(n))
                if max_chunk >= 0:
                    folders[folder.name] = RecordingFolder(folder.name, folder, max_chunk)
        return folders

    def frame_counts(self, frame_type: FrameType, suffix: str) -> list[int]:
        """Each chunk's frame count (camera 0), by chunk number; it decodes when a clip does not say."""
        frames: list[int] = []
        for chunk in range(self.chunks + 1):
            path: Path = self.path / make_chunk_name(0, chunk)
            if not path.is_file():
                path = self.path / make_file_name(0, frame_type, chunk, suffix)
            frames.append(StreamReader.frame_count(str(path)) if path.is_file() else 0)
        return frames

    def read_markers(self) -> list[float]:
        return read_in_out(self.path)

    def write_markers(self, in_out: list[float]) -> None:
        write_in_out(self.path, in_out)


@dataclass(frozen=True)
class Timeline:
    """A recording's chunks as one timeline of frames, and its markers on it: where the play starts (`start_chunk`,
    `start_offset`, inclusive) and ends (`end_chunk`, `end_offset`, exclusive)."""
    chunk_frames: list[int] = field(default_factory=list)
    start_chunk: int = 0
    start_offset: int = 0
    end_chunk: int = 0
    end_offset: int = 0

    @property
    def total(self) -> int:
        return sum(self.chunk_frames)

    @staticmethod
    def of(chunk_frames: list[int], in_out: list[float]) -> 'Timeline':
        """The timeline of `chunk_frames` with the markers `in_out`, fractions of its length in either order."""
        total: int = sum(chunk_frames)
        if total <= 0:
            return Timeline(chunk_frames)
        a, b = (max(0.0, min(1.0, marker)) for marker in in_out)
        start_global: int = min(max(round(min(a, b) * total), 0), total - 1)
        end_global: int = min(max(round(max(a, b) * total), start_global + 1), total)
        start_chunk, start_offset = locate(chunk_frames, start_global)
        end_chunk, end_offset = locate_end(chunk_frames, end_global)
        return Timeline(chunk_frames, start_chunk, start_offset, end_chunk, end_offset)

    def place(self, fraction: float) -> int:
        """The frame at `fraction` of the length, on the whole timeline."""
        return min(round(fraction * self.total), self.total - 1)

    def global_frame(self, chunk: int, frame_id: int) -> int:
        """A chunk's frame on the whole timeline; 0 for no chunk."""
        return sum(self.chunk_frames[:chunk]) + frame_id if chunk >= 0 else 0

    def within_markers(self, place: int) -> bool:
        return self.global_frame(self.start_chunk, self.start_offset) <= place < self.global_frame(self.end_chunk, self.end_offset)


def locate(chunk_frames: list[int], global_frame: int) -> tuple[int, int]:
    """An (inclusive) frame on the whole timeline as (chunk, frame in the chunk)."""
    acc: int = 0
    for i, n in enumerate(chunk_frames):
        if global_frame < acc + n:
            return i, global_frame - acc
        acc += n
    last: int = len(chunk_frames) - 1
    return last, max(0, chunk_frames[last] - 1)


def locate_end(chunk_frames: list[int], end_global: int) -> tuple[int, int]:
    """An (exclusive) end on the whole timeline as (chunk, exclusive end in the chunk)."""
    acc: int = 0
    for i, n in enumerate(chunk_frames):
        if end_global <= acc + n:
            return i, end_global - acc
        acc += n
    last: int = len(chunk_frames) - 1
    return last, chunk_frames[last]
