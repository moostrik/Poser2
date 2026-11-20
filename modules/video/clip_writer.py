"""ClipWriter: raw frames piped into one ffmpeg encoder process, written to an mp4."""

import logging
import os
import subprocess
from enum import Enum
from typing import BinaryIO

import numpy as np

from .encoder import H264_OPTIONS
from .hardware import ClipCodec, Hardware, encoder_name
from .process import error_text, pipe, popen

logger = logging.getLogger(__name__)


MAX_COMMENT: int = 30000    # characters; a Windows command line holds 32767


class ClipInput(Enum):
    """The pixel format of the frames a ClipWriter is given."""
    BGR24   = 'bgr24'       # (height, width, 3) uint8
    NV12    = 'nv12'        # (height × 3/2, width) uint8: Y, then Cb Cr at half size; BT.709 video range
    YUV444P = 'yuv444p'     # (height × 3, width) uint8: Y, Cb and Cr planes at full size; BT.709 video range
    P010    = 'p010le'      # as NV12 in uint16, 10 bits in the high bits
    XV30    = 'xv30le'      # (height, width) uint32: Cb, Y, Cr in bits 0-9, 10-19, 20-29; 4:4:4 at 10 bits


# the rows a YUV input's array holds per picture row, its sample type, and what hevc_nvenc is given: xv30le is
# unpacked by ffmpeg to 16-bit planes on the CPU, which costs only the encode's time
_YUV_ROWS: dict[ClipInput, float] = {ClipInput.NV12: 1.5, ClipInput.YUV444P: 3.0, ClipInput.P010: 1.5,
                                     ClipInput.XV30: 1.0}
_YUV_DTYPE: dict[ClipInput, type[np.unsignedinteger]] = {ClipInput.NV12: np.uint8, ClipInput.YUV444P: np.uint8,
                                                        ClipInput.P010: np.uint16, ClipInput.XV30: np.uint32}
_HEVC_PIX_FMT: dict[ClipInput, str] = {ClipInput.NV12: 'nv12', ClipInput.YUV444P: 'yuv444p', ClipInput.P010: 'p010le',
                                       ClipInput.XV30: 'yuv444p16le'}


HEVC_QUALITY: int = 18      # NVENC constant quality, lower is better


def _hevc_args(quality: int) -> list[str]:
    """hevc_nvenc at constant quality, its slowest preset with a full-resolution second pass and spatial and temporal
    adaptive quantization: the encode runs after a take, so its time goes to quality. Tagged hvc1 so Windows and
    QuickTime play it."""
    return ['-c:v', 'hevc_nvenc', '-preset', 'p7', '-tune', 'hq', '-multipass', 'fullres', '-rc', 'vbr',
            '-cq', str(quality), '-b:v', '0', '-spatial-aq', '1', '-temporal-aq', '1', '-tag:v', 'hvc1']


# what a YUV input is, told to the file: BT.709 in video range
_BT709_TAGS: list[str] = ['-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                          '-color_range', 'tv']


class ClipWriter:
    """Writes frames of one size to an mp4: bgr24 or a YUV input in, h264 on `hardware` or HEVC on the GPU out, through
    a pipe of `PIPE_SIZE`; the process starts with the first frame. comment, set before that frame, goes into the
    file's comment tag."""

    def __init__(self, path: str, fps: float, hardware: Hardware, clip_input: ClipInput = ClipInput.BGR24,
                 codec: ClipCodec = ClipCodec.H264, quality: int = HEVC_QUALITY) -> None:
        """`quality` is the HEVC constant quality; the h264 encoders keep their own."""
        if codec == ClipCodec.HEVC and hardware != Hardware.GPU:
            raise ValueError(f'HEVC is written on the GPU only, not on {hardware.value}')
        self.path: str = path
        self.fps: float = fps
        self.hardware: Hardware = hardware
        self.clip_input: ClipInput = clip_input
        self.codec: ClipCodec = codec
        self.quality: int = quality
        self.comment: str = ''
        self.frames: int = 0
        self.failed: bool = False
        self._size: tuple[int, int] = (0, 0)
        self._process: subprocess.Popen | None = None
        self._stdin: BinaryIO | None = None

    def _frame_size(self, image: np.ndarray) -> tuple[int, int]:
        """(width, height) of the picture in `image`; bgr24 is cropped to even, a YUV input must be even already."""
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
        if self.codec == ClipCodec.HEVC:
            pix_fmt = _HEVC_PIX_FMT.get(self.clip_input, 'nv12')
            output = [*_hevc_args(self.quality), '-pix_fmt', pix_fmt]
        else:
            codec = encoder_name(ClipCodec.H264, self.hardware)
            quality, pix_fmt = H264_OPTIONS[self.hardware]
            output = ['-c:v', codec, *quality, '-pix_fmt', pix_fmt]
        tags = _BT709_TAGS if self.clip_input in _YUV_ROWS else []
        args = ['-y', '-f', 'rawvideo', '-pix_fmt', self.clip_input.value, '-s', f'{width}x{height}',
                '-r', f'{self.fps:g}', '-i', 'pipe:0', *output, *tags, *metadata, self.path]
        read_fd, write_fd = pipe()
        try:
            self._process = popen(args, read_fd, subprocess.DEVNULL)
        finally:
            os.close(read_fd)       # the child has its own copy; ours must close for its read to reach the end
        self._stdin = os.fdopen(write_fd, 'wb', buffering=0)

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
            _, err = process.communicate(timeout=timeout)       # waits for the moov atom
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.failed = True
            return 'did not finish, killed'
        if process.returncode != 0:
            self.failed = True
        return error_text(err)

    def close(self) -> bool:
        """Finish the file. True when it is complete and has frames."""
        if self._process is None:
            return False
        err = self._end(timeout=30)
        if self.failed:
            logger.error('Encoder failed on %s: %s', self.path, err)
            return False
        return self.frames > 0
