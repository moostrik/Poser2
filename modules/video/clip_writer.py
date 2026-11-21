"""ClipWriter: raw frames piped into one ffmpeg encoder process, written to a file, and the encodes it is given: h264
for a clip, an HEVC master, a camera recording's stream."""

import logging
import os
import subprocess
from enum import Enum
from typing import BinaryIO

import numpy as np

from .encoder import H264_OPTIONS
from .hardware import ClipCodec, Hardware, encoder_name
from .process import ErrorReader, pipe, popen

logger = logging.getLogger(__name__)


MAX_COMMENT: int = 30000    # characters; a Windows command line holds 32767


class ClipInput(Enum):
    """The pixel format of the frames a ClipWriter is given."""
    BGR24   = 'bgr24'       # (height, width, 3) uint8
    GRAY    = 'gray'        # (height, width) uint8
    NV12    = 'nv12'        # (height × 3/2, width) uint8: Y, then Cb Cr at half size
    YUV444P = 'yuv444p'     # (height × 3, width) uint8: Y, Cb and Cr planes at full size
    P010    = 'p010le'      # as NV12 in uint16, 10 bits in the high bits
    P012    = 'p012le'      # as NV12 in uint16, 12 bits in the high bits: a camera's depth (depth_frame.py)
    XV30    = 'xv30le'      # (height, width) uint32: Cb, Y, Cr in bits 0-9, 10-19, 20-29; 4:4:4 at 10 bits


# the rows a YUV input's array holds per picture row, and its sample type
_YUV_ROWS: dict[ClipInput, float] = {ClipInput.NV12: 1.5, ClipInput.YUV444P: 3.0, ClipInput.P010: 1.5,
                                     ClipInput.P012: 1.5, ClipInput.XV30: 1.0}
_YUV_DTYPE: dict[ClipInput, type[np.unsignedinteger]] = {ClipInput.NV12: np.uint8, ClipInput.YUV444P: np.uint8,
                                                        ClipInput.P010: np.uint16, ClipInput.P012: np.uint16,
                                                        ClipInput.XV30: np.uint32}
# what hevc_nvenc is given for an input: xv30le is unpacked by ffmpeg to 16-bit planes on the CPU, which costs only
# the encode's time
_MASTER_PIX_FMT: dict[ClipInput, str] = {ClipInput.NV12: 'nv12', ClipInput.YUV444P: 'yuv444p',
                                         ClipInput.P010: 'p010le', ClipInput.XV30: 'yuv444p16le'}


# THE ENCODES: ffmpeg's output options, from the codec to the pixel format it is given

def h264_encode(hardware: Hardware) -> list[str]:
    """h264 on `hardware` (not AUTO) at its own constant quality."""
    quality, pix_fmt = H264_OPTIONS[hardware]
    return ['-c:v', encoder_name(ClipCodec.H264, hardware), *quality, '-pix_fmt', pix_fmt]


def master_encode(clip_input: ClipInput, quality: int) -> list[str]:
    """HEVC on the GPU at constant quality `quality`, from YUV planes in BT.709 video range: the slowest preset with a
    full-resolution second pass and spatial and temporal adaptive quantization, tagged hvc1 and BT.709 video range."""
    return ['-c:v', encoder_name(ClipCodec.HEVC, Hardware.GPU), '-preset', 'p7', '-tune', 'hq', '-multipass',
            'fullres', '-rc', 'vbr', '-cq', str(quality), '-b:v', '0', '-spatial-aq', '1', '-temporal-aq', '1',
            '-tag:v', 'hvc1', '-pix_fmt', _MASTER_PIX_FMT[clip_input],
            '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-color_range', 'tv']


def stream_encode(codec: ClipCodec, hardware: Hardware, pix_fmt: str, quality: int = 0, keyframe_interval: int = 0,
                  b_frames: bool = True) -> list[str]:
    """A camera recording's stream, encoded live: `codec` on `hardware` (not AUTO) into `pix_fmt`, at the encoder's
    constant quality `quality` (0: its default), a keyframe every `keyframe_interval` frames exactly (0: as the encoder
    likes), and without B-frames when asked."""
    args = ['-c:v', encoder_name(codec, hardware), '-pix_fmt', pix_fmt]
    if quality > 0:
        args += ['-q:v', str(quality)]
    if not b_frames:
        args += ['-bf', '0']
    if keyframe_interval > 0:
        args += ['-g', str(keyframe_interval), '-keyint_min', str(keyframe_interval)]     # no early keyframes
    return args


class ClipWriter:
    """Writes frames of one size and pixel format (`clip_input`) to a file with the output options `encode`, through a
    pipe of `PIPE_SIZE`; the process starts with the first frame, and its error lines are read while it writes.
    comment, set before that frame, goes into the file's comment tag."""

    def __init__(self, path: str, fps: float, encode: list[str], clip_input: ClipInput = ClipInput.BGR24) -> None:
        self.path: str = path
        self.fps: float = fps
        self.encode: list[str] = encode
        self.clip_input: ClipInput = clip_input
        self.comment: str = ''
        self.frames: int = 0
        self.failed: bool = False
        self._size: tuple[int, int] = (0, 0)
        self._process: subprocess.Popen | None = None
        self._stdin: BinaryIO | None = None
        self._errors: ErrorReader | None = None

    def _frame_size(self, image: np.ndarray) -> tuple[int, int]:
        """(width, height) of the picture in `image`; bgr24 and gray are cropped to even, a YUV input must be even
        already."""
        rows = _YUV_ROWS.get(self.clip_input)
        if rows is not None:
            width, height = image.shape[1], round(image.shape[0] / rows)
            dtype = _YUV_DTYPE[self.clip_input]
            if (image.ndim != 2 or width % 2 or height % 2 or round(height * rows) != image.shape[0]
                    or image.dtype != dtype):
                raise ValueError(f'{self.clip_input.value} frame of shape {image.shape} {image.dtype} is not '
                                 f'(even height × {rows:g}, even width) {np.dtype(dtype).name}')
            return width, height
        return image.shape[1] - image.shape[1] % 2, image.shape[0] - image.shape[0] % 2

    def add(self, image: np.ndarray) -> None:
        if self.failed:
            return
        width, height = self._frame_size(image)
        if self._process is None:
            self._size = (width, height)
            self._start()
        elif (width, height) != self._size:
            raise ValueError(f'frame {width}x{height} does not match the clip {self._size[0]}x{self._size[1]}')
        assert self._stdin is not None
        if self.clip_input in _YUV_ROWS:
            data = memoryview(np.ascontiguousarray(image)).cast('B')
        else:
            data = memoryview(np.ascontiguousarray(image[:height, :width])).cast('B')
        try:
            written = 0
            while written < len(data):      # an unbuffered pipe takes what fits in one write
                written += self._stdin.write(data[written:]) or 0
        except (BrokenPipeError, OSError) as e:
            self.failed = True
            logger.error('Encoder died on %s: %s %s', self.path, e, self._end(timeout=30))
            return
        self.frames += 1

    def _start(self) -> None:
        width, height = self._size
        metadata: list[str] = []
        if len(self.comment) > MAX_COMMENT:
            logger.warning('Comment of %s is too long (%d characters), left out', self.path, len(self.comment))
        elif self.comment:
            metadata = ['-metadata', f'comment={self.comment}']
        args = ['-y', '-f', 'rawvideo', '-pix_fmt', self.clip_input.value, '-s', f'{width}x{height}',
                '-r', f'{self.fps:g}', '-i', 'pipe:0', *self.encode, *metadata, self.path]
        read_fd, write_fd = pipe()
        try:
            self._process = popen(args, read_fd, subprocess.DEVNULL)
        finally:
            os.close(read_fd)       # the child has its own copy; ours must close for its read to reach the end
        self._stdin = os.fdopen(write_fd, 'wb', buffering=0)
        self._errors = ErrorReader(self._process)

    def _end(self, timeout: float) -> str:
        """Close the pipe and wait for ffmpeg; its error output. Kills it after `timeout` s."""
        process, self._process = self._process, None
        if self._stdin is not None:
            try:
                self._stdin.close()
            except OSError:
                pass
            self._stdin = None
        if process is None:
            return ''
        try:
            process.wait(timeout=timeout)       # waits for the moov atom
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            self.failed = True
            return 'did not finish, killed'
        if process.returncode != 0:
            self.failed = True
        return self._errors.text() if self._errors is not None else ''

    def close(self) -> bool:
        """Finish the file. True when it is complete and has frames."""
        if self._process is None:
            return False
        err = self._end(timeout=30)
        if self.failed:
            logger.error('Encoder failed on %s: %s', self.path, err)
            return False
        return self.frames > 0
