"""One ffmpeg encoder process fed raw frames from a queue: the writer of one camera recording stream.

It writes every frame it is given, in order: a stream of a recording must hold every frame its other streams and its
metadata hold, so whatever is dropped is dropped whole, by the recorder, before any part of it is written. The bytes
it holds are readable (`queued_bytes`), for the recorder's bound. An encoder that dies (an error, a full disk) ends
the writer: it says so once (`failed`), drops what is queued, and `stop` and `split` return instead of waiting."""

import logging
from threading import Lock, Thread
from queue import Empty, Queue
from time import sleep

import ffmpeg
import numpy as np

from .depth_frame import DEPTH_PIX_FMT, DEPTH_QUALITY, depth_to_p012

logger = logging.getLogger(__name__)


class StreamWriter:
    def __init__(self, vcodec: str, keyframe_interval: int = 0, quality: int = 0) -> None:
        """``quality``: the encoder's constant quality, lower is better; 0 is its default."""
        self.output_file: str = ''
        self.fps: float = 30.0
        self.is_recording: bool = False
        self.is_receiving: bool = False
        self.frames: Queue[tuple[np.ndarray, bool]] = Queue()
        self.thread: Thread | None = None
        self.vcodec: str = vcodec
        self.keyframe_interval: int = keyframe_interval
        self.quality: int = quality
        self._lock: Lock = Lock()
        self._queued_bytes: int = 0
        self._failed: bool = False

    @property
    def queued_bytes(self) -> int:
        """The bytes of the frames queued for the encoder."""
        with self._lock:
            return self._queued_bytes

    @property
    def failed(self) -> bool:
        """The encoder of the file being written died."""
        return self._failed

    def start(self, output_file: str, fps: float) -> None:
        if self.is_recording:
            return
        self.fps = fps
        self.output_file = output_file
        self._failed = False
        self.is_recording = True        # before the thread: a stop right after the start finds it recording
        self.is_receiving = True
        self.thread = Thread(target=self._record, daemon=True, name='StreamWriter')
        self.thread.start()

    def stop(self) -> None:
        if not self.is_recording:
            return
        self.is_receiving = False
        self._drain()
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()

    def split(self, output_file: str, fps: float) -> None:
        """Close this file and go on in `output_file`. What is queued goes into this file, so a
        caller that stops feeding first gets the same frames in every stream's chunk."""
        if not self.is_recording:
            return
        self._drain()
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()
        self.start(output_file, fps)

    def _drain(self) -> None:
        """Wait until the encoder has taken what is queued, or has died."""
        while not self.frames.empty() and self.thread is not None and self.thread.is_alive():
            sleep(0.01)

    def submit_frame(self, frame: np.ndarray, nv12: bool = False) -> None:
        """Queue one frame: BGR (H, W, 3), gray (H, W), uint16 depth in millimetres (H, W), or with
        `nv12` packed NV12 (H * 3 / 2, W), which needs saying because it is 2-D too."""
        if not self.is_receiving:
            return
        with self._lock:
            self._queued_bytes += frame.nbytes
        self.frames.put((frame, nv12))

    def _record(self) -> None:
        process = None
        try:
            while self.is_recording:
                try:
                    frame, nv12 = self.frames.get(timeout=.1)
                except Empty:
                    continue
                with self._lock:
                    self._queued_bytes -= frame.nbytes

                depth: bool = frame.dtype == np.uint16
                if process is None:
                    process = self._open(frame, nv12, depth)
                if depth:
                    process.stdin.write(depth_to_p012(frame))
                else:
                    process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).data)   # the frame's own buffer when it is uint8 and contiguous
        except (OSError, ValueError) as e:
            self._failed = True
            self.is_receiving = False
            logger.error('Encoder of %s ended: %s; the stream is dropped from here', self.output_file, e)
            self._clear()
        finally:
            if process is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
                process.wait()

    def _open(self, frame: np.ndarray, nv12: bool, depth: bool):
        width: int = frame.shape[1]
        height: int = frame.shape[0] * 2 // 3 if nv12 else frame.shape[0]
        pix_fmt: str = DEPTH_PIX_FMT if depth else 'nv12' if nv12 else 'gray' if len(frame.shape) == 2 else 'bgr24'
        output_kwargs = dict(pix_fmt=DEPTH_PIX_FMT if depth else 'yuv420p', vcodec=self.vcodec, r=self.fps)
        if depth:
            output_kwargs['q:v'] = DEPTH_QUALITY
            output_kwargs['bf'] = 0     # measured: with B-frames a seek in a merged chunk gave depth about 100 mm off, why is not settled
        elif self.quality > 0:
            output_kwargs['q:v'] = self.quality
        if self.keyframe_interval > 0:
            output_kwargs['g'] = self.keyframe_interval          # GOP size
            output_kwargs['keyint_min'] = self.keyframe_interval  # force exact spacing (no early IDR skip)
        return (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}', r=self.fps)
            .output(self.output_file, **output_kwargs)
            .overwrite_output()
            .global_args('-loglevel', 'quiet')
            .run_async(pipe_stdin=True)
        )

    def _clear(self) -> None:
        while True:
            try:
                frame, _ = self.frames.get_nowait()
            except Empty:
                return
            with self._lock:
                self._queued_bytes -= frame.nbytes
