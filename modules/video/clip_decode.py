"""A whole clip decoded into one array through ffmpeg, on a background process."""

import os
import subprocess
import threading
from typing import BinaryIO

import numpy as np

from .hardware import Hardware, hwaccel_args
from .probe import probe
from .process import big_pipe, error_text, popen


DECODE_THREADS: int = 2     # a background job: the players' render and loaders must keep their cores
DECODE_TIMEOUT: float = 60.0    # s a clip may take to decode (it takes a few); then ffmpeg is ended


def _read_exactly(file: BinaryIO, size: int) -> bytes:
    """Up to size bytes from an unbuffered file, which returns what the pipe holds per read; short only at its end."""
    parts: list[bytes] = []
    while size > 0:
        data = file.read(size)
        if not data:
            break
        parts.append(data)
        size -= len(data)
    return b''.join(parts)


def decode_clip(path: str, size: tuple[int, int], decoder: Hardware = Hardware.CPU) -> tuple[np.ndarray, float]:
    """The whole clip as a (frames, height, width, 3) bgr uint8 array scaled to size (width, height), and its fps."""
    info = probe(path)
    frames, fps = info.frames, info.fps
    width, height = size
    frame_bytes = width * height * 3
    args = ['-nostdin', '-threads', str(DECODE_THREADS), *hwaccel_args(decoder), '-i', path,
            '-filter_threads', str(DECODE_THREADS), '-vf', f'scale={width}:{height}',
            '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    out, write_fd = big_pipe()
    try:
        process = popen(args, subprocess.DEVNULL, write_fd, background=True)
    finally:
        os.close(write_fd)      # the child has its own copy; ours must close for the read to reach its end
    assert process.stderr is not None
    # stderr is read while the frames come in: a damaged clip's line per frame would fill its pipe
    # and stop ffmpeg while we wait for frames
    errors: list[bytes] = []
    stderr = process.stderr
    reader = threading.Thread(target=lambda: errors.append(stderr.read()), name='ffmpeg stderr', daemon=True)
    reader.start()
    # a decode that hangs is ended: the read below then reaches its end and the clip is skipped
    killer = threading.Timer(DECODE_TIMEOUT, process.kill)
    killer.daemon = True
    killer.start()
    try:
        if frames > 0:
            # fill a preallocated array, no second copy of the 200 MB
            array = np.empty((frames, height, width, 3), np.uint8)
            view = memoryview(array).cast('B')
            got = 0
            while got < len(view):
                n = out.readinto(view[got:])
                if not n:
                    break
                got += n
            while out.read(1 << 20):     # drain frames beyond the count, so ffmpeg can end
                pass
        else:
            # the file does not say how many frames: read to the end and stack
            chunks: list[bytes] = []
            while True:
                data = _read_exactly(out, frame_bytes)
                if len(data) < frame_bytes:
                    break
                chunks.append(data)
            got = len(chunks) * frame_bytes
            array = np.frombuffer(b''.join(chunks), np.uint8).reshape(-1, height, width, 3)
        out.close()
        process.wait(timeout=30)        # stdout is at its end, so ffmpeg is finishing
        reader.join(timeout=5)
        process.stderr.close()
    finally:
        killer.cancel()
        if process.poll() is None:
            process.kill()      # something went wrong here: no ffmpeg is left behind
    err: bytes = b''.join(errors)
    if process.returncode != 0:
        raise RuntimeError(f'decode failed on {path}: {error_text(err) or f"exit code {process.returncode}"}')
    if array.shape[0] == 0 or got < array.nbytes:
        raise RuntimeError(f'short read on {path}: {got // frame_bytes} of {array.shape[0]} frames')
    return array, fps if fps > 0 else 15.0
