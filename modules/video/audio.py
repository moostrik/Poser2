"""Audio files read and transcoded through ffmpeg."""

import subprocess

import numpy as np

from .process import error_text, popen


def transcode_aac(src: str, dst: str, bitrate: str = '128k') -> None:
    """Write `dst` (an .m4a) from any audio file: AAC, which every phone decodes (Safari refuses some MP3s),
    at about a third of the MP3's size. Raises when ffmpeg fails."""
    args = ['-y', '-i', src, '-vn', '-c:a', 'aac', '-b:a', bitrate, '-movflags', '+faststart', dst]
    process = popen(args, subprocess.DEVNULL, subprocess.DEVNULL)
    try:
        _, err = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError(f'ffmpeg did not finish transcoding {src}')
    if process.returncode != 0:
        raise RuntimeError(f'ffmpeg could not transcode {src}: {error_text(err)}')


def read_audio(path: str, offset: float, seconds: float, rate: int = 48000) -> np.ndarray:
    """`seconds` of an audio file from `offset` s as (n, 2) float32 at `rate`; silence pads a file that
    ends first. Raises when ffmpeg fails."""
    args = ['-ss', f'{max(0.0, offset):.3f}', '-t', f'{seconds:.3f}', '-i', path,
            '-f', 'f32le', '-ac', '2', '-ar', str(rate), 'pipe:1']
    process = popen(args, subprocess.DEVNULL, subprocess.PIPE)
    try:
        out, err = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError(f'ffmpeg did not finish reading {path}')
    if process.returncode != 0:
        raise RuntimeError(f'ffmpeg could not read {path}: {error_text(err)}')
    audio: np.ndarray = np.frombuffer(out[:len(out) - len(out) % 8], np.float32).reshape(-1, 2)
    wanted: int = round(seconds * rate)
    if audio.shape[0] < wanted:
        audio = np.concatenate([audio, np.zeros((wanted - audio.shape[0], 2), np.float32)])
    return audio
