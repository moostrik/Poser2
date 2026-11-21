"""A whole clip decoded into one array through ffmpeg, on a background process."""

import os
import subprocess
import threading

import numpy as np

from .hardware import Hardware, hwaccel_args
from .probe import probe
from .process import ErrorReader, big_pipe, popen, read_exactly


DECODE_THREADS: int = 2     # a background job: the players' render and loaders must keep their cores
DECODE_TIMEOUT: float = 60.0    # s a clip may take to decode (it takes a few); then ffmpeg is ended


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
    errors = ErrorReader(process)
    # a decode that hangs is ended: the read below then reaches its end and the clip is skipped
    killer = threading.Timer(DECODE_TIMEOUT, process.kill)
    killer.daemon = True
    killer.start()
    try:
        if frames > 0:
            # fill a preallocated array, no second copy of the 200 MB
            array = np.empty((frames, height, width, 3), np.uint8)
            got = read_exactly(out, memoryview(array).cast('B'))
            while out.read(1 << 20):     # drain frames beyond the count, so ffmpeg can end
                pass
        else:
            # the file does not say how many frames: read to the end and stack
            chunks: list[bytearray] = []
            while True:
                data = bytearray(frame_bytes)
                if read_exactly(out, memoryview(data)) < frame_bytes:
                    break
                chunks.append(data)
            got = len(chunks) * frame_bytes
            array = np.frombuffer(b''.join(chunks), np.uint8).reshape(-1, height, width, 3)
        out.close()
        process.wait(timeout=30)        # stdout is at its end, so ffmpeg is finishing
    finally:
        killer.cancel()
        if process.poll() is None:
            process.kill()      # something went wrong here: no ffmpeg is left behind
    err: str = errors.text()
    if process.returncode != 0:
        raise RuntimeError(f'decode failed on {path}: {err or f"exit code {process.returncode}"}')
    if array.shape[0] == 0 or got < array.nbytes:
        raise RuntimeError(f'short read on {path}: {got // frame_bytes} of {array.shape[0]} frames')
    return array, fps if fps > 0 else 15.0
