import ffmpeg
import numpy as np
from threading import Thread, Event, Lock
from typing import Callable
from enum import Enum
from cv2 import cvtColor, COLOR_RGB2BGR
import time


from ..camera.definitions import FrameType

import logging
logger = logging.getLogger(__name__)

class HwAccelerationType(Enum):
    CPU =   0
    GPU =   1
    iGPU =  2

HwaccelString: dict[HwAccelerationType, str] = {
    HwAccelerationType.GPU:  'd3d12va',
    HwAccelerationType.iGPU: 'd3d12va'
}

HwaccelDeviceString: dict[HwAccelerationType, str] = {
    HwAccelerationType.GPU:  '0',
    HwAccelerationType.iGPU: '1'
}

# PlayerCallback = Callable[[int, FrameType, np.ndarray, ], None]
EndCallback = Callable[[int], None]

class StreamReader:
    def __init__(self, cam_id: int, frameType: FrameType, frameCallback,
                 hw_acceleration_type: str = '', hw_acceleration_device: str = '', fps: float = 0.0) -> None:
        self.frame_callback = frameCallback

        self.cam_id: int = cam_id
        self.frame_type: FrameType = frameType
        self.hw_acceleration_type: str = hw_acceleration_type
        self.hw_acceleration_device: str = hw_acceleration_device

        self.ffmpeg_process = None
        self.bytes_per_frame: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.frame_rate: float = fps

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
        self.stop()
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
        if self._play_thread is not None:
            self.stop_event.set()

    def join(self) -> None:
        if self._load_thread is not None:
            self._load_thread.join()
            self._load_thread = None
        if self._play_thread is not None:
            self._play_thread.join()
            self._play_thread = None

    def _load(self, video_file: str, chunk_id: int, start_frame: int = 0, end_frame: int | None = None) -> None:
        try:
            self.frame_width, self.frame_height, frame_rate = self._get_video_dimensions(video_file)
            if self.frame_rate == 0.0:
                self.frame_rate = frame_rate
        except Exception as e:
            logger.error('Error getting video dimensions:', e)
            self.ffmpeg_process = None
            return

        pix_fmt: str = 'rgb24' if self.frame_type == FrameType.VIDEO else 'gray'
        self.bytes_per_frame: int = self.frame_width * self.frame_height * (3 if self.frame_type == FrameType.VIDEO else 1)

        ffmpeg_process = None
        try:
            if self.hw_acceleration_type == '' or self.hw_acceleration_device == '':
                ffmpeg_process = (
                    ffmpeg
                    .input(video_file)
                    .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                    .global_args('-loglevel', 'quiet')
                    .run_async(pipe_stdout=True)
                )
            else:
                ffmpeg_process = (
                    ffmpeg
                    .input(video_file, hwaccel=self.hw_acceleration_type, hwaccel_device=self.hw_acceleration_device)
                    .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                    .global_args('-loglevel', 'quiet')
                    .run_async(pipe_stdout=True)
                )

        except ffmpeg.Error as e:
            logger.error('Error loading:', e)
            self.ffmpeg_process = None

        with self._load_lock:
            self.ffmpeg_process = ffmpeg_process
            self.chunk_id = chunk_id
            self.start_frame = start_frame
            self.end_frame = end_frame

    def _play(self) -> None:
        if self.ffmpeg_process is None:
            logger.warning('video not loaded')
            return
        frame_interval: float = 1.0 / self.frame_rate

        start_frame: int = self.start_frame
        end_frame: int | None = self.end_frame
        frame_count: int = 0

        # Skip frames before the start of the requested range (decode & discard,
        # no callback, no pacing). Bounded by one chunk.
        while frame_count < start_frame and not self.stop_event.is_set():
            skip_bytes = self.ffmpeg_process.stdout.read(self.bytes_per_frame)
            if not skip_bytes:
                break
            frame_count += 1

        while not self.stop_event.is_set():
            if end_frame is not None and frame_count >= end_frame:
                break
            start_time: float = time.time()
            try:
                in_bytes = self.ffmpeg_process.stdout.read(self.bytes_per_frame)
                if not in_bytes:
                    # print ('End of stream', self.frame_type)
                    break
            except ffmpeg.Error as e:
                logger.error('Error reading frame:', e)
                break

            if self.frame_type == FrameType.VIDEO:
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([self.frame_height, self.frame_width, 3])
                frame = cvtColor(frame, COLOR_RGB2BGR)
            else:
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([self.frame_height, self.frame_width])

            self.frame_callback(self.cam_id, self.frame_type, frame, self.chunk_id, frame_count)
            frame_count += 1

            elapsed_time: float = time.time() - start_time
            sleep_time: float = max(0, frame_interval - elapsed_time)
            time.sleep(sleep_time)
        self.ffmpeg_process.stdout.close()
        self.ffmpeg_process.terminate()
        self.ffmpeg_process = None

    @staticmethod
    def frame_count(video_file: str) -> int:
        """Return the number of frames in *video_file*.

        Uses ``nb_frames`` when the container provides it (e.g. .mp4), else
        derives it from ``duration * fps``. Falls back to a full ``-count_frames``
        decode for raw elementary streams (e.g. .hevc) that expose neither.
        """
        try:
            probe = ffmpeg.probe(video_file)
        except ffmpeg.Error as e:
            logger.error('Error probing frame count: %s', e)
            return 0
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if video_stream is None:
            return 0

        nb_frames = video_stream.get('nb_frames')
        if nb_frames is not None and str(nb_frames).isdigit() and int(nb_frames) > 0:
            return int(nb_frames)

        duration = video_stream.get('duration') or probe.get('format', {}).get('duration')
        frame_rate_str: str = video_stream.get('r_frame_rate', '30/1')
        num, denom = map(int, frame_rate_str.split('/'))
        fps: float = num / denom if denom != 0 else 0.0
        if duration is not None and fps > 0.0:
            return round(float(duration) * fps)

        # Last resort: count decoded frames (slow, but reliable for raw streams).
        try:
            probe = ffmpeg.probe(video_file, count_frames=None, select_streams='v:0')
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
            if video_stream is not None:
                read_frames = video_stream.get('nb_read_frames')
                if read_frames is not None and str(read_frames).isdigit():
                    return int(read_frames)
        except ffmpeg.Error as e:
            logger.error('Error counting frames: %s', e)
        return 0

    def _get_video_dimensions(self, video_file: str) -> tuple:
        probe = ffmpeg.probe(video_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise Exception('No video stream found for file', video_file)
        width = int(video_stream['width'])
        height = int(video_stream['height'])

        frame_rate_str: str = video_stream.get('r_frame_rate', '30/1')
        num, denom = map(int, frame_rate_str.split('/'))
        frame_rate: float = num / denom if denom != 0 else 0

        return width, height, frame_rate