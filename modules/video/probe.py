"""What ffprobe says about a clip: one video stream's size, rate and exact frame count, and the clip's comment."""

import json
from dataclasses import dataclass
from fractions import Fraction

from .process import FFPROBE, run


@dataclass(frozen=True)
class StreamInfo:
    """A video stream of a clip: its size (px), its frame rate (frames/s, 0 when the file does not say), its frames,
    and the clip's comment tag (ClipWriter.comment), None when it has none."""
    width: int
    height: int
    fps: float
    frames: int
    comment: str | None


def _ffprobe(args: list[str], path: str) -> dict:
    try:
        out = run([*args, '-of', 'json', path], timeout=60, command=FFPROBE)
    except RuntimeError as e:
        raise RuntimeError(f'could not probe {path}: {e}') from e
    return json.loads(out)


def probe(path: str, stream: int = 0) -> StreamInfo:
    """Video stream `stream` of the clip at `path`. Its frames are the count the container stores (mp4), else its
    packets counted (mkv, raw streams), which reads the file without decoding it. Raises when ffprobe fails or the
    clip has no such stream."""
    select = ['-select_streams', f'v:{stream}']
    info = _ffprobe([*select, '-show_entries', 'stream=nb_frames,r_frame_rate,width,height:format_tags=comment'], path)
    streams = info.get('streams', [])
    if not streams:
        raise RuntimeError(f'no video stream {stream} in {path}')
    found = streams[0]
    stored = str(found.get('nb_frames', ''))
    if stored.isdigit() and int(stored) > 0:
        frames = int(stored)
    else:
        counted = _ffprobe([*select, '-count_packets', '-show_entries', 'stream=nb_read_packets'], path)
        packets = str((counted.get('streams') or [{}])[0].get('nb_read_packets', ''))
        frames = int(packets) if packets.isdigit() else 0
    fps = float(Fraction(found.get('r_frame_rate', '0/1'))) if found.get('r_frame_rate', '0/1') != '0/0' else 0.0
    comment = info.get('format', {}).get('tags', {}).get('comment')
    return StreamInfo(int(found['width']), int(found['height']), fps, frames, comment)
