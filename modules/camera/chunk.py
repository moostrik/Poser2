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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from modules.utils import Rect
from modules.video import popen

from .frame import DepthAlignment, FrameReadings
from .tracklet import CameraTracklet, TrackingStatus

logger = logging.getLogger(__name__)

METADATA_MIMETYPE: str = 'application/x-ndjson'
METADATA_SUFFIX: str = '.jsonl'


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


@dataclass(frozen=True)
class FrameMetadata:
    """What a chunk keeps of one frame beside its images: its capture time, the camera's readings,
    the frame's latency (its arrival minus its capture, s), the camera's tracklets of it as the
    camera gave them and their latency. None where the frame had none: `tracklets` None is a
    frame no tracklets came for, an empty list a frame with no player."""
    timestamp: float
    readings: FrameReadings | None = None
    video_latency: float | None = None
    tracklets: list[CameraTracklet] | None = None
    tracklet_latency: float | None = None


@dataclass(frozen=True)
class ChunkMetadata:
    info: ChunkInfo | None
    frames: list[FrameMetadata]         # by frame number


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
    return json.dumps({'frame': number, 'timestamp': frame.timestamp, 'readings': readings, 'video_latency': frame.video_latency,
                       'tracklets': tracklets, 'tracklet_latency': frame.tracklet_latency}) + '\n'


def _frame(values: dict, cam_id: int) -> FrameMetadata:
    timestamp: float = values['timestamp']
    readings: dict | None = values.get('readings')
    tracklets: list[dict] | None = values.get('tracklets')
    return FrameMetadata(
        timestamp,
        FrameReadings(**readings) if readings is not None else None,
        values.get('video_latency'),
        [CameraTracklet(cam_id=cam_id, id=t['id'], status=TrackingStatus[t['status']], roi=Rect(*t['roi']), age=t['age'], timestamp=timestamp,
                        keypoints=np.array(t['keypoints'], dtype=np.float32) if t['keypoints'] is not None else None)
         for t in tracklets] if tracklets is not None else None,
        values.get('tracklet_latency'))


def parse_metadata(text: str, cam_id: int = 0) -> ChunkMetadata:
    """The info line and one `FrameMetadata` per following line, the tracklets of camera `cam_id`. A
    line that does not parse ends the list: a chunk cut off by a crash stops there, and what came
    before it is good."""
    lines: list[str] = text.splitlines()
    info: ChunkInfo | None = None
    frames: list[FrameMetadata] = []
    try:
        if lines:
            info = chunk_info(json.loads(lines[0]))
        for line in lines[1:]:
            frames.append(_frame(json.loads(line), cam_id))
    except (ValueError, TypeError, KeyError) as e:
        logger.warning('Chunk metadata cut short: %s', e)
    return ChunkMetadata(info, frames)


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
