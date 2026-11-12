"""One recorded chunk as one file: color, depth when recorded, and what the camera reported, in a `.mkv`.

The streams are recorded apart, in the temp folder, and merged by stream copy when the chunk ends,
which re-encodes nothing and takes moments. The fixed info of the chunk goes into the file's tags,
where any tool can read it; the per-frame lines go in as an attachment. Stream 0 is color, 1 depth.

Nothing in a chunk is straightened or mirrored: the frames and the tracklets are as the camera
delivered them, and `flip_h` says the frames are still to be mirrored, as `CameraFrame.flip_h` does.
One line per frame, in the frames' order, so a line's place is its frame's number.
"""

import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import TypeAlias, overload

import numpy as np

from modules.utils import Rect
from modules.video import popen

from ...frame import DepthAlignment, FrameReadings, FrameType
from ...tracklet import CameraTracklet, TrackingStatus

logger = logging.getLogger(__name__)

METADATA_MIMETYPE: str = 'application/x-ndjson'
METADATA_SUFFIX: str = '.jsonl'
FRAME_TYPE_LABELS: dict[FrameType, str] = {     # a stream's file while the chunk is recorded
    FrameType.VIDEO: 'C',
    FrameType.DEPTH: 'D',
}
METADATA_LABEL: str = 'M'
DEPTH_SUFFIX: str = '.mkv'


def make_file_name(cam_id: int, frame_type: FrameType, chunk: int, suffix: str) -> str:
    """A stream's file while the chunk is recorded, and a color-only clip of before."""
    return f"{cam_id}_{FRAME_TYPE_LABELS[frame_type]}_{chunk:03d}{suffix}"


def make_chunk_name(cam_id: int, chunk: int) -> str:
    """The one file a chunk ends as."""
    return f"{cam_id}_{chunk:03d}.mkv"


def make_folder_name(name: str = "") -> str:
    """A recording's folder: its start time, and its name."""
    folder = time.strftime("%Y%m%d-%H%M%S")
    if name:
        folder += f"_{name}"
    return folder


@dataclass(frozen=True)
class ChunkInfo:
    """What stays the same through a chunk: the first line of its metadata, and its tags.
    `depth_alignment` is the depth's, as the camera gave it; None for depth recorded on the video's
    view already. `depth` says the chunk holds a depth stream."""
    width: int
    height: int
    intrinsics: list[list[float]] | None
    flip_h: bool
    fps: float
    depth_alignment: DepthAlignment | None = None
    depth: bool = True


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


def info_line(info: ChunkInfo) -> str:
    alignment: dict | None = asdict(info.depth_alignment) if info.depth_alignment is not None else None
    return json.dumps({'width': info.width, 'height': info.height, 'intrinsics': info.intrinsics,
                       'flip_h': info.flip_h, 'fps': info.fps, 'depth_alignment': alignment, 'depth': info.depth}) + '\n'


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
    """A chunk's frame lines, each parsed into its `FrameMetadata` when it is first read and kept: a chunk of a
    minute is 3000 lines, some 40 ms of parsing at once **(measured: 8.3 ms for 597 lines with poses)**, which
    spread over the frames read costs a frame nothing."""

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


def merge_chunk(video: Path, depth: Path | None, metadata: Path, output: Path) -> bool:
    """`video`, and `depth` when the chunk has it, by stream copy into the `.mkv` at `output`,
    `metadata` attached and its info line in the tags. The parts are removed only when the file is
    there; False leaves them."""
    info: ChunkInfo | None = parse_metadata(metadata.read_text(encoding='utf-8')).info
    tags: list[str] = []
    if info is not None:
        alignment: dict | None = asdict(info.depth_alignment) if info.depth_alignment is not None else None
        for key, value in (('width', info.width), ('height', info.height), ('fps', info.fps), ('flip_h', info.flip_h),
                           ('intrinsics', json.dumps(info.intrinsics)), ('depth_alignment', json.dumps(alignment)), ('depth', info.depth)):
            tags += ['-metadata', f'{key}={value}']
    inputs: list[str] = ['-i', str(video)] + (['-i', str(depth)] if depth is not None else [])
    maps: list[str] = ['-map', '0:v:0'] + (['-map', '1:v:0'] if depth is not None else [])
    done: bool = _run(['-y', *inputs, *maps, '-c', 'copy',
                       '-attach', str(metadata), '-metadata:s:t:0', f'mimetype={METADATA_MIMETYPE}', *tags, str(output)])
    if done:
        for part in (video, depth, metadata):
            if part is not None:
                part.unlink(missing_ok=True)
    return done


def read_metadata(path: Path, cam_id: int = 0) -> ChunkMetadata:
    """The metadata attached to a chunk's `.mkv`; empty when it has none."""
    with tempfile.TemporaryDirectory() as folder:
        dump: Path = Path(folder) / 'metadata.jsonl'
        _run(['-y', '-dump_attachment:t:0', str(dump), '-i', str(path), '-t', '0', '-f', 'null', '-'])
        if not dump.is_file():
            return ChunkMetadata(None, [])
        return parse_metadata(dump.read_text(encoding='utf-8'), cam_id)


def finish_unmerged(temp_path: Path, output_path: Path) -> None:
    """Merge what a crash left in the temp folder: a chunk with its color, its depth when it has one,
    and its metadata. A stream cut off mid-write has no playable file, so its merge fails and its
    parts stay."""
    if not temp_path.is_dir():
        return
    for folder in (entry for entry in temp_path.iterdir() if entry.is_dir()):
        for metadata in folder.glob(f'*_M_*{METADATA_SUFFIX}'):
            cam, _, chunk = metadata.stem.split('_')
            video: Path | None = next(folder.glob(f'{cam}_C_{chunk}.*'), None)
            depth: Path | None = next(folder.glob(f'{cam}_D_{chunk}.*'), None)
            info: ChunkInfo | None = parse_metadata(metadata.read_text(encoding='utf-8')).info
            if video is not None and (depth is not None or (info is not None and not info.depth)):
                target: Path = output_path / folder.name
                target.mkdir(parents=True, exist_ok=True)
                logger.info('Finishing chunk %s_%s of %s', cam, chunk, folder.name)
                merge_chunk(video, depth, metadata, target / f'{cam}_{chunk}.mkv')
        if not any(folder.iterdir()):
            shutil.rmtree(folder, ignore_errors=True)
