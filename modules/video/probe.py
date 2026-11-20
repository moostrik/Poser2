"""What ffprobe says about a clip: its first video stream, its comment."""

import json
import subprocess
from fractions import Fraction

from .process import FFPROBE, error_text, popen


def probe(path: str) -> tuple[int, float, int, int]:
    """(frames, fps, width, height) of the first video stream; frames is 0 when the file does not say."""
    args = ['-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,r_frame_rate,width,height',
            '-of', 'json', path]
    process = popen(args, subprocess.DEVNULL, subprocess.PIPE, command=FFPROBE)
    out, err = process.communicate(timeout=20)
    if process.returncode != 0:
        raise RuntimeError(f'ffprobe failed on {path}: {error_text(err)}')
    streams = json.loads(out).get('streams', [])
    if not streams:
        raise RuntimeError(f'no video stream in {path}')
    stream = streams[0]
    try:
        frames = int(stream.get('nb_frames', 0))
    except ValueError:
        frames = 0
    fps = float(Fraction(stream.get('r_frame_rate', '0/1')))
    return frames, fps, int(stream['width']), int(stream['height'])


def read_comment(path: str) -> str | None:
    """The file's comment tag (ClipWriter.comment), or None when it has none."""
    args = ['-show_entries', 'format_tags=comment', '-of', 'json', path]
    process = popen(args, subprocess.DEVNULL, subprocess.PIPE, command=FFPROBE)
    out, err = process.communicate(timeout=20)
    if process.returncode != 0:
        raise RuntimeError(f'ffprobe failed on {path}: {error_text(err)}')
    return json.loads(out).get('format', {}).get('tags', {}).get('comment')
