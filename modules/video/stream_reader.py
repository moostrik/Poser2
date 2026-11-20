"""One ffmpeg decoder process read frame by frame: the reader of one camera recording stream, paced
on `perf_counter` and handing each frame to a callback with its index. It paces on its own at `fps`
from its start, or on a `FrameSchedule` it is given, which several readers can share."""

import io
import os
import subprocess
import numpy as np
from threading import Thread, Event, Lock
from typing import Callable, Protocol
import time

from modules.utils import wait_until

from .depth_frame import DEPTH_PIX_FMT, depth_frame_bytes, p012_to_depth
from .hardware import Hardware, hwaccel_args
from .probe import probe
from .process import big_pipe, popen

import logging
logger = logging.getLogger(__name__)

# (cam_id, frame_type, frame, chunk_id, frame_id): a stream knows which camera and kind it is
# reading for and says so with every frame, the kind being whatever the caller passed in.
StreamFrameCallback = Callable[[int, object, np.ndarray, int, int], None]


class FrameSchedule(Protocol):
    """When the readers of a recording hand their frames over, on `perf_counter`."""
    def moment(self, cam_id: int, frame_id: int) -> float | None:
        """When frame `frame_id` of camera `cam_id` is due; None where the schedule does not know it."""
        ...

    def catch_up(self, cam_id: int, frame_id: int, now: float) -> float:
        """A reader found its frame overdue at `now`: every frame of the schedule comes later, as far as that frame
        is still behind on the schedule as it stands, so two readers late for one stall move it once. The
        frame's moment after it."""
        ...


class StreamReader:
    def __init__(self, cam_id: int, frame_type: object, frame_callback: StreamFrameCallback, color: bool,
                 decoder: Hardware = Hardware.CPU, fps: float = 0.0, stream: int = 0, depth: bool = False,
                 schedule: FrameSchedule | None = None, nv12: bool = False) -> None:
        """``color``: decode to BGR, or with ``nv12`` to packed NV12 (H * 3 / 2, W), half the bytes, as a camera
        delivers it; otherwise to gray, so a mono pipeline gets the luma straight from ffmpeg, saving the colour
        conversion and two thirds of the pipe. ``stream``: which video stream of the file. ``depth``: it holds
        depth, delivered as uint16 millimetres. ``schedule``: when each frame is due, in place of every 1 / ``fps``
        from the reader's start, for the frames it knows."""
        self.frame_callback: StreamFrameCallback = frame_callback
        self.schedule: FrameSchedule | None = schedule
        self.nv12: bool = nv12 and color and not depth

        self.cam_id: int = cam_id
        self.frame_type: object = frame_type
        self.color: bool = color
        self.stream: int = stream
        self.depth: bool = depth
        self.channels: int = 1
        self.decoder: Hardware = decoder

        self.ffmpeg_process: subprocess.Popen | None = None
        self._frames_in: io.FileIO | None = None     # the decoder's frames, through the 8 MB pipe
        self.bytes_per_frame: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.fps: float = fps

        self._load_lock: Lock = Lock()
        self._load_thread: Thread | None = None

        self._play_thread: Thread | None = None
        self.stop_event = Event()

        self.chunk_id: int
        self.start_frame: int = 0
        self.end_frame: int | None = None

    def is_loaded(self) -> bool:
        with self._load_lock:
            return self.ffmpeg_process is not None

    def is_loading(self) -> bool:
        return self._load_thread is not None and self._load_thread.is_alive()

    def is_playing(self) -> bool:
        return self._play_thread is not None and self._play_thread.is_alive()

    def is_stopped(self) -> bool:
        return not self.is_loading() and not self.is_playing()

    def load(self, video_file: str, chunk_id: int, start_frame: int = 0, end_frame: int | None = None) -> None:
        self._load_thread = Thread(target=self._load, args=(video_file, chunk_id, start_frame, end_frame))
        self._load_thread.start()

    def play(self) -> None:
        if not self.is_loaded():
            logger.warning('video not loaded')
            return
        if self._play_thread is not None:
            logger.warning('video already playing')
            return

        self._play_thread = Thread(target=self._play)
        self._play_thread.start()

    def stop(self) -> None:
        """End the reader for good. One that never played ends its decoder here: nothing else would,
        and a reader still loading ends it when the load is done. One that plays has its decoder ended
        here too, so a read waiting on a decoder that stalled returns and the reader's thread ends."""
        self.stop_event.set()
        if self._play_thread is None:
            self._close()
            return
        with self._load_lock:
            process = self.ffmpeg_process
        if process is not None:
            process.terminate()

    def join(self) -> None:
        if self._load_thread is not None:
            self._load_thread.join()
            self._load_thread = None
        if self._play_thread is not None:
            self._play_thread.join()
            self._play_thread = None

    def _load(self, video_file: str, chunk_id: int, start_frame: int = 0, end_frame: int | None = None) -> None:
        try:
            info = probe(video_file, self.stream)
            self.frame_width, self.frame_height, fps = info.width, info.height, info.fps
            if self.fps == 0.0:
                self.fps = fps
            # Half a frame early: the seek takes the first frame at or after it, and stamps are rounded.
            seek: list[str] = ['-ss', f'{(start_frame - 0.5) / fps}'] if start_frame > 0 else []
        except Exception as e:
            logger.error('Error getting video dimensions of %s: %s', video_file, e)
            self.ffmpeg_process = None
            return

        pix_fmt: str = DEPTH_PIX_FMT if self.depth else 'nv12' if self.nv12 else 'bgr24' if self.color else 'gray'     # BGR matches getCvFrame
        self.channels = 3 if self.color and not self.depth and not self.nv12 else 1
        self.bytes_per_frame: int = (depth_frame_bytes(self.frame_width, self.frame_height) if self.depth
                                     else self.frame_width * self.frame_height * 3 // 2 if self.nv12
                                     else self.frame_width * self.frame_height * self.channels)

        args: list[str] = ['-nostdin', *seek, *hwaccel_args(self.decoder), '-i', video_file, '-map', f'0:v:{self.stream}',
                           '-f', 'rawvideo', '-pix_fmt', pix_fmt, 'pipe:1']
        ffmpeg_process: subprocess.Popen | None = None
        frames_in: io.FileIO | None = None
        try:
            frames_in, write_fd = big_pipe()
            try:
                ffmpeg_process = popen(args, subprocess.DEVNULL, write_fd)
            finally:
                os.close(write_fd)      # the child has its own copy; ours must close for the read to reach the end
            Thread(target=self._log_errors, args=(ffmpeg_process, video_file), name='StreamReader errors',
                   daemon=True).start()
        except OSError as e:
            logger.error('Error loading %s: %s', video_file, e)
            if frames_in is not None:
                frames_in.close()
            ffmpeg_process, frames_in = None, None

        with self._load_lock:
            self.ffmpeg_process = ffmpeg_process
            self._frames_in = frames_in
            self.chunk_id = chunk_id
            self.start_frame = start_frame
            self.end_frame = end_frame
        if self.stop_event.is_set():
            self._close()

    def _log_errors(self, process: subprocess.Popen, video_file: str) -> None:
        """ffmpeg's error lines, read as they come so its pipe never fills, and logged unless the reader was
        stopped (an ended decoder may say so)."""
        assert process.stderr is not None
        for line in process.stderr:
            if not self.stop_event.is_set():
                logger.error('Decoder of %s: %s', video_file, line.decode(errors='replace').strip())

    def _close(self) -> None:
        with self._load_lock:
            process, self.ffmpeg_process = self.ffmpeg_process, None
            frames_in, self._frames_in = self._frames_in, None
        if process is not None:
            process.terminate()
            process.wait()
        if frames_in is not None:
            frames_in.close()

    @staticmethod
    def _read_frame(frames_in: io.FileIO, frame: bytearray) -> bool:
        """Fill `frame` from the pipe, which hands over what it holds per read; False at the stream's end."""
        view = memoryview(frame)
        got = 0
        while got < len(view):
            n = frames_in.readinto(view[got:])
            if not n:
                return False
            got += n
        return True

    def _play(self) -> None:
        frames_in = self._frames_in
        if self.ffmpeg_process is None or frames_in is None:
            logger.warning('video not loaded')
            return
        frame_interval: float = 1.0 / self.fps

        end_frame: int | None = self.end_frame
        frame_count: int = self.start_frame     # the decoder starts there: `_load` seeks

        # Absolute schedule on perf_counter (the light clock's pattern): per-iteration sleeps on
        # the coarse wall clock quantize every frame by up to ~16 ms and let errors accumulate.
        next_time: float = time.perf_counter()
        while not self.stop_event.is_set():
            if end_frame is not None and frame_count >= end_frame:
                break
            try:
                # A bytearray, not bytes: an array over bytes is read-only, and torch warns on it
                in_bytes = bytearray(self.bytes_per_frame)
                if not self._read_frame(frames_in, in_bytes):
                    break       # the stream's end, or its decoder ended by `stop`
            except (OSError, ValueError) as e:
                if not self.stop_event.is_set():
                    logger.error('Error reading frame: %s', e)
                break

            if self.depth:
                frame: np.ndarray = p012_to_depth(in_bytes, self.frame_width, self.frame_height)
            elif self.nv12:
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([self.frame_height * 3 // 2, self.frame_width])
            elif self.channels == 3:
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([self.frame_height, self.frame_width, 3])
            else:
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([self.frame_height, self.frame_width])

            due: float | None = self.schedule.moment(self.cam_id, frame_count) if self.schedule is not None else None
            if due is not None and self.schedule is not None:
                now: float = time.perf_counter()
                if now - due > frame_interval:
                    due = self.schedule.catch_up(self.cam_id, frame_count, now)     # behind: every reader comes later, together
                due = max(due, next_time)           # never before the frame before it
                if wait_until(due, self.stop_event.wait):
                    break               # stopped while it waited
                next_time = due

            self.frame_callback(self.cam_id, self.frame_type, frame, self.chunk_id, frame_count)
            frame_count += 1

            if due is not None:
                continue
            next_time += frame_interval
            remaining: float = next_time - time.perf_counter()
            if remaining > 0:
                wait_until(next_time, self.stop_event.wait)
            elif -remaining > frame_interval:
                next_time = time.perf_counter()   # resync after a stall rather than bursting catch-up frames
        self._close()
