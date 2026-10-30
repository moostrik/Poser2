"""One recorded chunk as one file: colour, depth and what the camera reported, in a `.mkv`.

The streams are recorded apart, in the temp folder, and merged by stream copy when the chunk ends,
which re-encodes nothing and takes moments. The fixed info of the chunk goes into the file's tags,
where any tool can read it; the per-frame lines go in as an attachment. Stream 0 is colour, 1 depth.
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from modules.video import popen

from .frame import FrameReadings

logger = logging.getLogger(__name__)

METADATA_MIMETYPE: str = 'application/x-ndjson'
METADATA_SUFFIX: str = '.jsonl'


@dataclass(frozen=True)
class ChunkInfo:
    """What stays the same through a chunk: the first line of its metadata, and its tags."""
    width: int
    height: int
    intrinsics: list[list[float]] | None
    flip_h: bool
    fps: float


@dataclass(frozen=True)
class ChunkMetadata:
    info: ChunkInfo | None
    readings: list[FrameReadings]       # by frame number


def info_line(info: ChunkInfo) -> str:
    return json.dumps({'width': info.width, 'height': info.height, 'intrinsics': info.intrinsics,
                       'flip_h': info.flip_h, 'fps': info.fps}) + '\n'


def readings_line(frame: int, timestamp: float, readings: FrameReadings) -> str:
    return json.dumps({'frame': frame, 'timestamp': timestamp, 'tilt': readings.tilt, 'roll': readings.roll,
                       'exposure': readings.exposure, 'iso': readings.iso}) + '\n'


def parse_metadata(text: str) -> ChunkMetadata:
    """The info line and one `FrameReadings` per following line. A line that does not parse ends
    the list: a chunk cut off by a crash stops there, and what came before it is good."""
    lines: list[str] = text.splitlines()
    info: ChunkInfo | None = None
    readings: list[FrameReadings] = []
    try:
        if lines:
            info = ChunkInfo(**json.loads(lines[0]))
        for line in lines[1:]:
            values: dict = json.loads(line)
            readings.append(FrameReadings(values['tilt'], values['roll'], values['exposure'], values['iso']))
    except (ValueError, TypeError, KeyError) as e:
        logger.warning('Chunk metadata cut short: %s', e)
    return ChunkMetadata(info, readings)


def _run(args: list[str]) -> bool:
    process: subprocess.Popen = popen(args, subprocess.DEVNULL, subprocess.DEVNULL, background=True)
    _, errors = process.communicate()
    if process.returncode != 0:
        logger.error('ffmpeg failed: %s', errors.decode(errors='replace').strip())
        return False
    return True


def merge_chunk(video: Path, depth: Path, metadata: Path, output: Path) -> bool:
    """`video` and `depth` by stream copy into the `.mkv` at `output`, `metadata` attached and its
    info line in the tags. The parts are removed only when the file is there; False leaves them."""
    info: ChunkInfo | None = parse_metadata(metadata.read_text(encoding='utf-8')).info
    tags: list[str] = []
    if info is not None:
        for key, value in (('width', info.width), ('height', info.height), ('fps', info.fps), ('flip_h', info.flip_h),
                           ('intrinsics', json.dumps(info.intrinsics))):
            tags += ['-metadata', f'{key}={value}']
    done: bool = _run(['-y', '-i', str(video), '-i', str(depth), '-map', '0:v:0', '-map', '1:v:0', '-c', 'copy',
                       '-attach', str(metadata), '-metadata:s:t:0', f'mimetype={METADATA_MIMETYPE}', *tags, str(output)])
    if done:
        for part in (video, depth, metadata):
            part.unlink(missing_ok=True)
    return done


def read_metadata(path: Path) -> ChunkMetadata:
    """The metadata attached to a chunk's `.mkv`; empty when it has none."""
    with tempfile.TemporaryDirectory() as folder:
        dump: Path = Path(folder) / 'metadata.jsonl'
        _run(['-y', '-dump_attachment:t:0', str(dump), '-i', str(path), '-t', '0', '-f', 'null', '-'])
        if not dump.is_file():
            return ChunkMetadata(None, [])
        return parse_metadata(dump.read_text(encoding='utf-8'))


def finish_unmerged(temp_path: Path, output_path: Path) -> None:
    """Merge what a crash left in the temp folder: a chunk with both streams and its metadata. A
    stream cut off mid-write has no playable file, so its merge fails and its parts stay."""
    if not temp_path.is_dir():
        return
    for folder in (entry for entry in temp_path.iterdir() if entry.is_dir()):
        for metadata in folder.glob(f'*_M_*{METADATA_SUFFIX}'):
            cam, _, chunk = metadata.stem.split('_')
            video: Path = next(folder.glob(f'{cam}_C_{chunk}.*'), folder / 'missing')
            depth: Path = next(folder.glob(f'{cam}_D_{chunk}.*'), folder / 'missing')
            if video.is_file() and depth.is_file():
                target: Path = output_path / folder.name
                target.mkdir(parents=True, exist_ok=True)
                logger.info('Finishing chunk %s_%s of %s', cam, chunk, folder.name)
                merge_chunk(video, depth, metadata, target / f'{cam}_{chunk}.mkv')
        if not any(folder.iterdir()):
            shutil.rmtree(folder, ignore_errors=True)
