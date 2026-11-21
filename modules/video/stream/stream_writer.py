"""A ClipWriter fed raw frames from a queue on its own thread: the writer of one camera recording stream.

It writes every frame it is given, in order: a stream of a recording must hold every frame its other streams and its
metadata hold, so whatever is dropped is dropped whole, by the recorder, before any part of it is written. The bytes
it holds are readable (`queued_bytes`), for the recorder's bound. An encoder that dies (an error, a full disk) ends
the writer: it says so once (`failed`), drops what is queued, and `stop` and `split` return instead of waiting."""

import logging
from threading import Event, Lock, Thread
from queue import Queue

import numpy as np

from ..clip_writer import ClipInput, ClipWriter, stream_encode
from ..hardware import ClipCodec, Hardware
from .depth_frame import DEPTH_PIX_FMT, DEPTH_QUALITY, depth_to_p012

logger = logging.getLogger(__name__)


class StreamWriter:
    def __init__(self, codec: ClipCodec, hardware: Hardware, keyframe_interval: int = 0, quality: int = 0,
                 b_frames: bool = True) -> None:
        """``codec`` on ``hardware``; ``quality``: the encoder's constant quality, lower is better; 0 is its default;
        ``b_frames`` off for a stream that seeks with the depth. Depth takes its own quality and no B-frames."""
        self.output_file: str = ''
        self.fps: float = 30.0
        self.is_recording: Event = Event()
        self.is_receiving: Event = Event()
        self.frames: Queue[tuple[np.ndarray, bool] | None] = Queue()     # None: the end of the file
        self.thread: Thread | None = None
        self.codec: ClipCodec = codec
        self.hardware: Hardware = hardware
        self.keyframe_interval: int = keyframe_interval
        self.quality: int = quality
        self.b_frames: bool = b_frames
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
        if self.is_recording.is_set():
            return
        self.fps = fps
        self.output_file = output_file
        self._failed = False
        self.is_recording.set()         # before the thread: a stop right after the start finds it recording
        self.is_receiving.set()
        self.thread = Thread(target=self._record, daemon=True, name='StreamWriter')
        self.thread.start()

    def stop(self) -> None:
        if not self.is_recording.is_set():
            return
        self.is_receiving.clear()
        self._finish()

    def split(self, output_file: str, fps: float) -> None:
        """Close this file and go on in `output_file`. What is queued goes into this file, so a
        caller that stops feeding first gets the same frames in every stream's chunk."""
        if not self.is_recording.is_set():
            return
        self._finish()
        self.start(output_file, fps)

    def _finish(self) -> None:
        """End the file after what is queued: the end marker goes on the queue, and once the encoder has taken all of
        it, or dropped it as it died, its thread ends."""
        self.frames.put(None)
        self.frames.join()
        self.is_recording.clear()
        if self.thread is not None:
            self.thread.join()

    def submit_frame(self, frame: np.ndarray, nv12: bool = False) -> None:
        """Queue one frame: BGR (H, W, 3), gray (H, W), uint16 depth in millimeters (H, W), or with
        `nv12` packed NV12 (H * 3 / 2, W), which needs saying because it is 2-D too."""
        if not self.is_receiving.is_set():
            return
        with self._lock:
            self._queued_bytes += frame.nbytes
        self.frames.put((frame, nv12))

    def _record(self) -> None:
        writer: ClipWriter | None = None
        try:
            while True:
                item = self.frames.get()
                try:
                    if item is None:
                        break
                    frame, nv12 = item
                    with self._lock:
                        self._queued_bytes -= frame.nbytes

                    depth: bool = frame.dtype == np.uint16
                    if writer is None:
                        writer = self._open(frame, nv12, depth)
                    if depth:
                        height, width = frame.shape
                        writer.add(np.frombuffer(depth_to_p012(frame), np.uint16).reshape(height * 3 // 2, width))
                    else:
                        writer.add(np.ascontiguousarray(frame, dtype=np.uint8))
                    if writer.failed:
                        raise OSError('the encoder died')
                finally:
                    self.frames.task_done()
        except Exception as e:
            self._failed = True
            self.is_receiving.clear()
            logger.error('Encoder of %s ended: %s; the stream is dropped from here', self.output_file, e)
        finally:
            if writer is not None:
                writer.close()
        if self._failed:
            self._clear()

    def _open(self, frame: np.ndarray, nv12: bool, depth: bool) -> ClipWriter:
        clip_input: ClipInput = (ClipInput.P012 if depth else ClipInput.NV12 if nv12
                                 else ClipInput.GRAY if frame.ndim == 2 else ClipInput.BGR24)
        # measured: with B-frames a seek in a merged chunk gave depth about 100 mm off, why is not settled
        encode: list[str] = stream_encode(self.codec, self.hardware, DEPTH_PIX_FMT if depth else 'yuv420p',
                                          DEPTH_QUALITY if depth else self.quality, self.keyframe_interval,
                                          b_frames=self.b_frames and not depth)
        return ClipWriter(self.output_file, self.fps, encode, clip_input)

    def _clear(self) -> None:
        """Drop what is queued up to the end marker, each marked done, so `stop` and `split` do not wait on a dead
        encoder."""
        while True:
            item = self.frames.get()
            self.frames.task_done()
            if item is None:
                return
            with self._lock:
                self._queued_bytes -= item[0].nbytes
