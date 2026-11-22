"""One recorded chunk: color, depth and its confidence when recorded, and what the camera reported, as one `.mkv`
or as the color `.mp4` with the metadata `.jsonl` beside it (`Container`).

The streams are recorded apart, in the temp folder, and for a `.mkv` merged by stream copy when the chunk ends,
which re-encodes nothing and takes moments. The fixed info of the chunk goes into the file's tags,
where any tool can read it; the per-frame lines go in as an attachment. The streams are `ChunkStream`'s: 0 color,
1 depth, 2 the depth's confidence.

Nothing in a chunk is straightened, turned or mirrored: the frames and the tracklets are as the camera
delivered them, and `rotate90`, `flip_v` and `flip_h` say the frames are still to be turned and flipped, as
`CameraFrame`'s do.
One line per frame, in the frames' order, so a line's place is its frame's number.
"""

import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from pathlib import Path
from collections.abc import Sequence
from typing import TypeAlias, overload

import numpy as np

from modules.camera import CameraTracklet, DepthAlignment, FrameReadings, FrameType, TrackingStatus
from modules.utils import Rect
from modules.video import popen

logger = logging.getLogger(__name__)

METADATA_MIMETYPE: str = 'application/x-ndjson'
METADATA_SUFFIX: str = '.jsonl'
METADATA_LABEL: str = 'M'
DEPTH_SUFFIX: str = '.mkv'      # the depth's and the confidence's file while the chunk is recorded


class ChunkStream(IntEnum):
    """The streams of a chunk, each its index in the `.mkv`: the color, the depth, and the depth's confidence."""
    VIDEO = 0
    DEPTH = 1
    CONFIDENCE = 2


CHUNK_STREAM_LABELS: dict[ChunkStream, str] = {     # a stream's letter in its file's name while the chunk is recorded
    ChunkStream.VIDEO: 'C',
    ChunkStream.DEPTH: 'D',
    ChunkStream.CONFIDENCE: 'F',
}


def chunk_streams(types: Sequence[FrameType]) -> list[ChunkStream]:
    """The streams of the camera's frame types, in their order: the color, and with depth the depth and its confidence."""
    return [ChunkStream.VIDEO] + ([ChunkStream.DEPTH, ChunkStream.CONFIDENCE] if FrameType.DEPTH in types else [])


class Container(Enum):
    """The file a chunk ends as: one `.mkv` of its streams with the metadata attached, or its color stream as an
    `.mp4` any player plays, with the metadata beside it and no depth."""
    MKV = '.mkv'
    MP4 = '.mp4'


def make_file_name(cam_id: int, stream: ChunkStream, chunk: int, suffix: str) -> str:
    """A stream's file while the chunk is recorded, and a color-only clip of before."""
    return f"{cam_id}_{CHUNK_STREAM_LABELS[stream]}_{chunk:03d}{suffix}"


def make_chunk_name(cam_id: int, chunk: int) -> str:
    """The one file a chunk ends as."""
    return f"{cam_id}_{chunk:03d}.mkv"


def make_metadata_name(cam_id: int, chunk: int) -> str:
    """A chunk's metadata file: while it is recorded, and beside an `.mp4` chunk's color clip."""
    return f"{cam_id}_{METADATA_LABEL}_{chunk:03d}{METADATA_SUFFIX}"


def parse_name(stem: str) -> tuple[int, int] | None:
    """The camera and the chunk number of a chunk's file name, `<cam>_<chunk>` or `<cam>_<label>_<chunk>`; None for
    another name."""
    parts: list[str] = stem.split('_')
    if len(parts) not in (2, 3) or not (parts[0].isdigit() and parts[-1].isdigit()):
        return None
    return int(parts[0]), int(parts[-1])


def make_folder_name(name: str = "", parent: Path | None = None) -> str:
    """A recording's folder: its start time, and its name; with a counter when `parent` has that folder already."""
    folder = time.strftime("%Y%m%d-%H%M%S")
    if name:
        folder += f"_{name}"
    if parent is None or not (parent / folder).exists():
        return folder
    count: int = 2
    while (parent / f"{folder}-{count}").exists():
        count += 1
    return f"{folder}-{count}"


@dataclass(frozen=True)
class ChunkInfo:
    """What stays the same through a chunk: the first line of its metadata, and its tags.
    `depth_alignment` is the depth's, as the camera gave it; None for depth recorded on the video's
    view already. `depth` says the chunk holds a depth stream, `confidence` that it holds the depth's
    confidence too."""
    width: int
    height: int
    intrinsics: list[list[float]] | None
    flip_h: bool
    fps: float
    depth_alignment: DepthAlignment | None = None
    depth: bool = True
    confidence: bool = False
    rotate90: bool = False
    flip_v: bool = False


# A frame's poses by track_id: the box they were found in (x, y, width, height, float32 (4,)) and the
# points in it (float32 (17, 3): x, y, score). The camera contract knows no pose; the app converts.
PoseArrays: TypeAlias = dict[int, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class FrameMetadata:
    """What a chunk keeps of one frame beside its images: its capture time, the camera's readings,
    the frame's latency (its arrival minus its capture, s), the camera's tracklets of it as the
    camera gave them and their latency, and the host's poses of it and their latency. None where
    the frame had none: `tracklets` None is a frame no tracklets came for, an empty list a frame with
    no player; `poses` None is a frame no poses came for."""
    timestamp: float
    readings: FrameReadings | None = None
    video_latency: float | None = None
    tracklets: list[CameraTracklet] | None = None
    tracklet_latency: float | None = None
    poses: PoseArrays | None = None
    pose_latency: float | None = None


@dataclass(frozen=True)
class ChunkMetadata:
    info: ChunkInfo | None
    frames: Sequence[FrameMetadata]     # by frame number


def info_values(info: ChunkInfo) -> dict:
    """The chunk's info by field name, the depth alignment as a dict: the info line's and the tags' values."""
    return asdict(info)


def info_line(info: ChunkInfo) -> str:
    return json.dumps(info_values(info)) + '\n'


def chunk_info(values: dict) -> ChunkInfo:
    """The info line's values as a `ChunkInfo`."""
    alignment: dict | None = values.pop('depth_alignment', None)
    depth_alignment: DepthAlignment | None = None
    if alignment is not None:
        depth_alignment = DepthAlignment(**{**alignment, 'color_size': tuple(alignment['color_size'])})
    return ChunkInfo(**values, depth_alignment=depth_alignment)


def frame_line(number: int, frame: FrameMetadata) -> str:
    readings: dict | None = asdict(frame.readings) if frame.readings is not None else None
    tracklets: list[dict] | None = None
    if frame.tracklets is not None:
        tracklets = [{'id': t.id, 'status': t.status.name, 'roi': [t.roi.x, t.roi.y, t.roi.width, t.roi.height], 'age': t.age,
                      'keypoints': t.keypoints.tolist() if t.keypoints is not None else None} for t in frame.tracklets]
    poses: dict[str, dict] | None = None
    if frame.poses is not None:
        poses = {str(track_id): {'box': _rounded(box), 'points': _rounded(points)} for track_id, (box, points) in frame.poses.items()}
    return json.dumps({'frame': number, 'timestamp': frame.timestamp, 'readings': readings, 'video_latency': frame.video_latency,
                       'tracklets': tracklets, 'tracklet_latency': frame.tracklet_latency,
                       'poses': poses, 'pose_latency': frame.pose_latency}) + '\n'


def _rounded(values: np.ndarray) -> list:
    """An array as lists at five decimals: a pose's coordinates and scores need no more."""
    return np.round(values.astype(np.float64), 5).tolist()


def _frame(values: dict, cam_id: int) -> FrameMetadata:
    timestamp: float = values['timestamp']
    readings: dict | None = values.get('readings')
    tracklets: list[dict] | None = values.get('tracklets')
    poses: dict[str, dict] | None = values.get('poses')
    return FrameMetadata(
        timestamp,
        FrameReadings(**readings) if readings is not None else None,
        values.get('video_latency'),
        [CameraTracklet(cam_id=cam_id, id=t['id'], status=TrackingStatus[t['status']], roi=Rect(*t['roi']), age=t['age'], timestamp=timestamp,
                        keypoints=np.array(t['keypoints'], dtype=np.float32) if t['keypoints'] is not None else None)
         for t in tracklets] if tracklets is not None else None,
        values.get('tracklet_latency'),
        {int(track_id): (np.array(p['box'], dtype=np.float32), np.array(p['points'], dtype=np.float32)) for track_id, p in poses.items()}
        if poses is not None else None,
        values.get('pose_latency'))


class FrameLines(Sequence[FrameMetadata]):
    """A chunk's frame lines, each parsed into its `FrameMetadata` when it is first read and kept."""

    def __init__(self, lines: list[str], cam_id: int) -> None:
        self._lines: list[str] = lines
        self._cam_id: int = cam_id
        self._frames: list[FrameMetadata | None] = [None] * len(lines)

    def __len__(self) -> int:
        return len(self._lines)

    @overload
    def __getitem__(self, index: int) -> FrameMetadata: ...
    @overload
    def __getitem__(self, index: slice) -> list[FrameMetadata]: ...

    def __getitem__(self, index: int | slice) -> FrameMetadata | list[FrameMetadata]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        frame: FrameMetadata | None = self._frames[index]
        if frame is None:
            frame = _frame(json.loads(self._lines[index]), self._cam_id)
            self._frames[index] = frame
        return frame


def parse_metadata(text: str, cam_id: int = 0) -> ChunkMetadata:
    """The info line and one `FrameMetadata` per following line, the tracklets of camera `cam_id`, each line
    parsed when it is first read (`FrameLines`). A last line that does not parse is left out: a chunk cut off
    by a crash stops there, and what came before it is good."""
    lines: list[str] = text.splitlines()
    info: ChunkInfo | None = None
    try:
        if lines:
            info = chunk_info(json.loads(lines[0]))
    except (ValueError, TypeError, KeyError) as e:
        logger.warning('Chunk metadata without its info line: %s', e)
        return ChunkMetadata(None, [])
    frame_lines: list[str] = lines[1:]
    while frame_lines:
        try:
            _frame(json.loads(frame_lines[-1]), cam_id)
            break
        except (ValueError, TypeError, KeyError) as e:
            logger.warning('Chunk metadata cut short: %s', e)
            frame_lines.pop()
    return ChunkMetadata(info, FrameLines(frame_lines, cam_id))


def _run(args: list[str]) -> bool:
    process: subprocess.Popen = popen(args, subprocess.DEVNULL, subprocess.DEVNULL, background=True)
    _, errors = process.communicate()
    if process.returncode != 0:
        logger.error('ffmpeg failed: %s', errors.decode(errors='replace').strip())
        return False
    return True


def merge_chunk(video: Path, depth: Path | None, confidence: Path | None, metadata: Path, output: Path) -> bool:
    """`video`, and `depth` and its `confidence` when the chunk has them, by stream copy into the `.mkv` at
    `output` in `ChunkStream`'s order, `metadata` attached and its info line in the tags. The parts are removed
    only when the file is there; False leaves them."""
    info: ChunkInfo | None = parse_metadata(metadata.read_text(encoding='utf-8')).info
    tags: list[str] = []
    if info is not None:
        for key, value in info_values(info).items():
            text: str = json.dumps(value) if value is None or isinstance(value, (list, dict)) else str(value)
            tags += ['-metadata', f'{key}={text}']
    streams: list[Path] = [video] + ([depth] if depth is not None else []) + \
        ([confidence] if depth is not None and confidence is not None else [])     # no confidence without its depth
    inputs: list[str] = [arg for path in streams for arg in ('-i', str(path))]
    maps: list[str] = [arg for i in range(len(streams)) for arg in ('-map', f'{i}:v:0')]
    done: bool = _run(['-y', *inputs, *maps, '-c', 'copy',
                       '-attach', str(metadata), '-metadata:s:t:0', f'mimetype={METADATA_MIMETYPE}', *tags, str(output)])
    if done:
        for part in (video, depth, confidence, metadata):
            if part is not None:
                part.unlink(missing_ok=True)
    return done


def keep_chunk(video: Path, metadata: Path, output_folder: Path) -> bool:
    """`video` and `metadata` moved as they are into `output_folder`: an `.mp4` chunk; False when a move failed."""
    try:
        shutil.move(video, output_folder / video.name)
        shutil.move(metadata, output_folder / metadata.name)
    except OSError as e:
        logger.error('Chunk %s not kept: %s', video.name, e)
        return False
    return True


def metadata_beside(clip: Path) -> Path:
    """The metadata file beside an `.mp4` chunk's color clip: `0_M_001.jsonl` beside `0_C_001.mp4`."""
    name: tuple[int, int] | None = parse_name(clip.stem)
    if name is None:
        raise ValueError(f'Not a chunk file: {clip.name}')
    return clip.with_name(make_metadata_name(*name))


def read_metadata(path: Path, cam_id: int = 0) -> ChunkMetadata:
    """The metadata of a chunk: attached to its `.mkv`, or beside its `.mp4` clip; empty when it has none."""
    if path.suffix != Container.MKV.value:
        beside: Path = metadata_beside(path)
        if not beside.is_file():
            return ChunkMetadata(None, [])
        return parse_metadata(beside.read_text(encoding='utf-8'), cam_id)
    with tempfile.TemporaryDirectory() as folder:
        dump: Path = Path(folder) / 'metadata.jsonl'
        _run(['-y', '-dump_attachment:t:0', str(dump), '-i', str(path), '-t', '0', '-f', 'null', '-'])
        if not dump.is_file():
            return ChunkMetadata(None, [])
        return parse_metadata(dump.read_text(encoding='utf-8'), cam_id)


def finish_unmerged(temp_path: Path, output_path: Path, container: Container = Container.MKV) -> None:
    """Finish what a crash left in the temp folder as `container`: a chunk with its color, its depth and its
    confidence when it has them, and its metadata. A stream cut off mid-write has no playable file, so its merge
    fails and its parts stay."""
    if not temp_path.is_dir():
        return
    for folder in (entry for entry in temp_path.iterdir() if entry.is_dir()):
        for metadata in folder.glob(f'*_{METADATA_LABEL}_*{METADATA_SUFFIX}'):
            try:
                _finish_unmerged_chunk(folder, metadata, output_path, container)
            except Exception:
                logger.exception('Chunk %s of %s not finished', metadata.name, folder.name)
        try:
            if not any(folder.iterdir()):
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            logger.exception('Temp folder %s not removed', folder)


def _finish_unmerged_chunk(folder: Path, metadata: Path, output_path: Path, container: Container) -> None:
    """One chunk a crash left in `folder`, by its metadata file, once all its streams are there."""
    name: tuple[int, int] | None = parse_name(metadata.stem)
    if name is None:
        return
    cam, chunk = name
    video, depth, confidence = (next(folder.glob(make_file_name(cam, stream, chunk, '.*')), None)
                                for stream in ChunkStream)
    info: ChunkInfo | None = parse_metadata(metadata.read_text(encoding='utf-8')).info
    with_depth: bool = info is None or info.depth
    with_confidence: bool = info is not None and info.confidence
    if video is None or (depth is None and with_depth) or (confidence is None and with_confidence):
        return
    target: Path = output_path / folder.name
    target.mkdir(parents=True, exist_ok=True)
    logger.info('Finishing chunk %s of %s', metadata.name, folder.name)
    if container == Container.MP4:
        keep_chunk(video, metadata, target)
    else:
        merge_chunk(video, depth, confidence, metadata, target / make_chunk_name(cam, chunk))
