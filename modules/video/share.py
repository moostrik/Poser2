"""The share clip: a clip re-encoded with overlays and a sound track, for a phone."""

import subprocess

import numpy as np

from .encoder import CODECS, Encoder
from .probe import probe
from .process import error_text, popen


# the share clip for a phone: lighter than the players' clip, it travels
_SHARE_QUALITY: dict[Encoder, list[str]] = {
    Encoder.QSV:   ['-global_quality', '27'],
    Encoder.NVENC: ['-preset', 'p4', '-rc', 'vbr', '-cq', '27', '-b:v', '0'],
    Encoder.CPU:   ['-crf', '25', '-preset', 'veryfast', '-threads', '4'],
}


def make_share(video: str, overlays: list[tuple[str, str]], pcm: np.ndarray, rate: int, encoder: Encoder, out: str) -> None:
    """Write `out`: the video re-encoded with the `overlays` (PNGs with alpha, each with its place as ffmpeg's
    overlay takes it, e.g. 'W-w-24:H-h-24') and `pcm` ((n, 2) float32 at `rate`) as its sound, cut to the
    shorter of the two; the video's comment tag is kept. Raises when ffmpeg fails."""
    codec, _, pix_fmt = CODECS[encoder]
    quality: list[str] = _SHARE_QUALITY[encoder]
    comment: str | None = probe(video).comment
    metadata: list[str] = ['-metadata', f'comment={comment}'] if comment else []
    inputs: list[str] = ['-i', video]
    chain: list[str] = []
    last: str = '[0:v]'
    for n, (png, place) in enumerate(overlays, start=1):
        inputs += ['-i', png]
        label: str = f'[v{n}]' if n < len(overlays) else ''
        chain.append(f'{last}[{n}:v]overlay={place}{label}')
        last = label
    filters: list[str] = ['-filter_complex', ';'.join(chain)] if chain else []
    audio_index: int = len(overlays) + 1
    args = ['-y', *inputs, '-f', 'f32le', '-ar', str(rate), '-ac', '2', '-i', 'pipe:0', *filters,
            '-map', f'{audio_index}:a:0', *([] if overlays else ['-map', '0:v:0']),
            '-c:v', codec, *quality, '-pix_fmt', pix_fmt, '-c:a', 'aac', '-b:a', '128k',
            '-shortest', '-movflags', '+faststart', *metadata, out]
    process = popen(args, subprocess.PIPE, subprocess.DEVNULL)
    try:
        _, err = process.communicate(input=np.ascontiguousarray(pcm, dtype=np.float32).tobytes(), timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError(f'ffmpeg did not finish the share clip of {video}')
    if process.returncode != 0:
        raise RuntimeError(f'ffmpeg could not make the share clip of {video}: {error_text(err)}')
