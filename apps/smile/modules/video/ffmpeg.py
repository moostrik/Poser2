"""ffmpeg and ffprobe as subprocesses: the grin clips are written and read through them.

The apps run under pythonw (scripts/start.bat), so every process is started with CREATE_NO_WINDOW
and without inheriting stdin or stderr, which are invalid handles there. stderr is a pipe that is
read when the process ends, so a failure has a message."""

import json
import logging
import os
import shutil
import subprocess
from enum import Enum
from fractions import Fraction
from typing import BinaryIO

import numpy as np

logger = logging.getLogger(__name__)

CREATE_NO_WINDOW: int = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
FFMPEG: str = 'ffmpeg'
FFPROBE: str = 'ffprobe'


class Encoder(Enum):
    AUTO  = 'Auto'
    QSV   = 'Intel Quick Sync'
    NVENC = 'Nvidia NVENC'
    CPU   = 'CPU (libx264)'


# codec, quality options, pixel format the codec takes
_CODECS: dict[Encoder, tuple[str, list[str], str]] = {
    Encoder.QSV:   ('h264_qsv',   ['-global_quality', '20'], 'nv12'),
    Encoder.NVENC: ('h264_nvenc', ['-preset', 'p4', '-rc', 'vbr', '-cq', '20', '-b:v', '0'], 'yuv420p'),
    Encoder.CPU:   ('libx264',    ['-crf', '18', '-preset', 'veryfast', '-threads', '4'], 'yuv420p'),
}
_AUTO_ORDER: list[Encoder] = [Encoder.QSV, Encoder.NVENC, Encoder.CPU]


def available() -> bool:
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


BELOW_NORMAL: int = getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)
DECODE_THREADS: int = 2     # a background job: the players' render and loaders must keep their cores


def popen(args: list[str], stdin, stdout, command: str = FFMPEG, background: bool = False) -> subprocess.Popen:
    """Start ffmpeg (or ffprobe) with the arguments after the log options; stderr is always a pipe.
    A background process runs below normal priority."""
    flags: int = CREATE_NO_WINDOW | (BELOW_NORMAL if background else 0)
    return subprocess.Popen([command, '-hide_banner', '-loglevel', 'error', *args],
                            stdin=stdin, stdout=stdout, stderr=subprocess.PIPE,
                            creationflags=flags)


PIPE_SIZE: int = 8 << 20     # the default Windows pipe holds 4 KB: reading a clip through it takes 6 times as long


def _big_pipe() -> tuple[BinaryIO, int]:
    """A pipe with a large buffer: (unbuffered read file, write descriptor to hand to the child)."""
    if os.name == 'nt':
        import _winapi
        import msvcrt
        read_handle, write_handle = _winapi.CreatePipe(None, PIPE_SIZE)
        read_fd = msvcrt.open_osfhandle(read_handle, os.O_RDONLY)
        write_fd = msvcrt.open_osfhandle(write_handle, 0)
    else:
        read_fd, write_fd = os.pipe()
    return os.fdopen(read_fd, 'rb', buffering=0), write_fd


def _stderr(output: bytes | None) -> str:
    return (output or b'').decode(errors='replace').strip()


# ENCODING

def encoder_works(encoder: Encoder) -> bool:
    """A one-frame trial encode: the encoder list alone does not prove the driver works."""
    codec, quality, pix_fmt = _CODECS[encoder]
    args = ['-nostdin', '-f', 'lavfi', '-i', 'color=c=gray:size=128x128:rate=15', '-frames:v', '1',
            '-c:v', codec, *quality, '-pix_fmt', pix_fmt, '-f', 'null', '-']
    try:
        process = popen(args, subprocess.DEVNULL, subprocess.DEVNULL)
        _, err = process.communicate(timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning('%s trial failed: %s', encoder.value, e)
        return False
    if process.returncode != 0:
        logger.info('%s not usable: %s', encoder.value, _stderr(err))
        return False
    return True


def pick_encoder(preference: Encoder = Encoder.AUTO) -> Encoder:
    """The encoder to use: the preferred one if its trial passes, else the first of Intel, Nvidia, CPU that does."""
    if not available():
        logger.error('ffmpeg or ffprobe not on the PATH: clips cannot be saved')
        return Encoder.CPU
    candidates = _AUTO_ORDER if preference == Encoder.AUTO else [preference, Encoder.CPU]
    for encoder in candidates:
        if encoder_works(encoder):
            if preference not in (Encoder.AUTO, encoder):
                logger.warning('%s not usable, using %s', preference.value, encoder.value)
            return encoder
    logger.error('No h264 encoder works, using %s anyway', Encoder.CPU.value)
    return Encoder.CPU


class ClipWriter:
    """Writes bgr24 frames of one size to an h264 mp4; the process starts with the first frame."""

    def __init__(self, path: str, fps: float, encoder: Encoder) -> None:
        self.path: str = path
        self.fps: float = fps
        self.encoder: Encoder = encoder
        self.frames: int = 0
        self.failed: bool = False
        self._size: tuple[int, int] = (0, 0)
        self._process: subprocess.Popen | None = None

    def add(self, image: np.ndarray) -> None:
        if self.failed:
            return
        height, width = image.shape[0] - image.shape[0] % 2, image.shape[1] - image.shape[1] % 2
        if self._process is None:
            self._size = (width, height)
            self._start()
        elif (width, height) != self._size:
            raise ValueError(f'frame {width}x{height} does not match the clip {self._size[0]}x{self._size[1]}')
        assert self._process is not None and self._process.stdin is not None
        data = np.ascontiguousarray(image[:height, :width]).tobytes()
        try:
            self._process.stdin.write(data)
        except (BrokenPipeError, OSError) as e:
            self.failed = True
            _, err = self._process.communicate()
            logger.error('Encoder died on %s: %s %s', self.path, e, _stderr(err))
            return
        self.frames += 1

    def _start(self) -> None:
        codec, quality, pix_fmt = _CODECS[self.encoder]
        width, height = self._size
        args = ['-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', f'{self.fps:g}',
                '-i', 'pipe:0', '-c:v', codec, *quality, '-pix_fmt', pix_fmt, self.path]
        self._process = popen(args, subprocess.PIPE, subprocess.DEVNULL)

    def close(self) -> bool:
        """Finish the file. True when it is complete and has frames."""
        if self._process is None:
            return False
        process, self._process = self._process, None
        try:
            _, err = process.communicate(timeout=30)     # closes stdin, waits for the moov atom
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            logger.error('Encoder did not finish %s, killed', self.path)
            return False
        if process.returncode != 0:
            logger.error('Encoder failed on %s: %s', self.path, _stderr(err))
            return False
        return not self.failed and self.frames > 0


# DECODING

def probe(path: str) -> tuple[int, float, int, int]:
    """(frames, fps, width, height) of the first video stream; frames is 0 when the file does not say."""
    args = ['-select_streams', 'v:0', '-show_entries', 'stream=nb_frames,r_frame_rate,width,height',
            '-of', 'json', path]
    process = popen(args, subprocess.DEVNULL, subprocess.PIPE, command=FFPROBE)
    out, err = process.communicate(timeout=20)
    if process.returncode != 0:
        raise RuntimeError(f'ffprobe failed on {path}: {_stderr(err)}')
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


def hwdecode_args(adapter: int | None) -> list[str]:
    """Input options for a D3D11 hardware decode on the DXGI adapter with that index (tested on the Intel iGPU)."""
    if adapter is None:
        return []
    return ['-hwaccel', 'd3d11va', '-hwaccel_device', str(adapter)]


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


def decode_clip(path: str, size: tuple[int, int], hwdecode: int | None = None) -> tuple[np.ndarray, float]:
    """The whole clip as a (frames, height, width, 3) bgr uint8 array scaled to size (width, height), and its fps."""
    frames, fps, _, _ = probe(path)
    width, height = size
    frame_bytes = width * height * 3
    args = ['-nostdin', '-threads', str(DECODE_THREADS), *hwdecode_args(hwdecode), '-i', path,
            '-filter_threads', str(DECODE_THREADS), '-vf', f'scale={width}:{height}',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    out, write_fd = _big_pipe()
    try:
        process = popen(args, subprocess.DEVNULL, write_fd, background=True)
    finally:
        os.close(write_fd)      # the child has its own copy; ours must close for the read to reach its end
    assert process.stderr is not None
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
    err = process.stderr.read()     # stdout is at its end, so ffmpeg is finishing
    out.close()
    process.stderr.close()
    process.wait(timeout=30)
    if process.returncode != 0:
        raise RuntimeError(f'decode failed on {path}: {_stderr(err)}')
    if array.shape[0] == 0 or got < array.nbytes:
        raise RuntimeError(f'short read on {path}: {got // frame_bytes} of {array.shape[0]} frames')
    return array, fps if fps > 0 else 15.0
