import ffmpeg
import numpy as np
from threading import Thread, Event, Lock
from typing import Callable
from enum import Enum
from cv2 import cvtColor, COLOR_RGB2BGR
import time


from modules.cam.depthcam.Definitions import FrameType

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

PlayerCallback = Callable[[int, FrameType, np.ndarray], None]
EndCallback = Callable[[int], None]

class FFmpegPlayer:
    def __init__(self, cam_id: int, frameType: FrameType,
                 frameCallback: PlayerCallback,
                 hw_acceleration_type: str = '', hw_acceleration_device: str = '') -> None:
        self.frame_callback: PlayerCallback = frameCallback

        self.cam_id: int = cam_id
        self.frame_type: FrameType = frameType
        self.hw_acceleration_type: str = hw_acceleration_type
        self.hw_acceleration_device: str = hw_acceleration_device

        self.ffmpeg_process = None
        self.bytes_per_frame: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

        self._load_lock: Lock = Lock()
        self._load_thread: Thread | None = None

        self._play_thread: Thread | None = None
        self.stop_event = Event()

        self.chunk_id: int

    def is_loaded(self) -> bool:
        with self._load_lock:
            return self.ffmpeg_process is not None

    def is_loading(self) -> bool:
        return self._load_thread is not None and self._load_thread.is_alive()

    def is_playing(self) -> bool:
        return self._play_thread is not None and self._play_thread.is_alive()

    def load(self, video_file: str, chunk_id: int) -> None:
        if self.is_loading():
            print('video already loading')
            return
        if self._load_thread is not None:
            self._load_thread.join()
        self._load_thread = Thread(target=self._load, args=(video_file, chunk_id))
        self._load_thread.start()

    def play(self) -> None:
        if not self.is_loaded():
            print('video not loaded')
            return
        if self._play_thread is not None:
            print('video already playing')
            return

        self._play_thread = Thread(target=self._play)
        self._play_thread.start()

    def stop(self) -> None:
        if self._load_thread is not None:
            self._load_thread.join()
        if self._play_thread is not None:
            self.stop_event.set()
            self._play_thread.join()
            self._play_thread = None

    def _load(self, video_file: str, chunk_id: int) -> None:
        try:
            self.frame_width, self.frame_height = self._get_video_dimensions(video_file)
        except Exception as e:
            print('Error getting video dimensions:', e)
            self.ffmpeg_process = None
            return

        pix_fmt: str = 'rgb24' if self.frame_type == FrameType.VIDEO else 'gray'
        self.bytes_per_frame: int = self.frame_width * self.frame_height * (3 if self.frame_type == FrameType.VIDEO else 1)

        ffmpeg_process = None
        try:
            if self.hw_acceleration_device == '' or self.hw_acceleration_type == '':
                ffmpeg_process = (
                    ffmpeg
                    .input(video_file, re=None)
                    .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                    .global_args('-loglevel', 'quiet')
                    .run_async(pipe_stdout=True)
                )
            else:
                ffmpeg_process = (
                    ffmpeg
                    .input(video_file, re=None, hwaccel=self.hw_acceleration_device, hwaccel_device=self.hw_acceleration_type)
                    .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
                    .global_args('-loglevel', 'quiet')
                    .run_async(pipe_stdout=True)
                )
        except ffmpeg.Error as e:
            print('Error loading:', e)
            self.ffmpeg_process = None

        with self._load_lock:
            self.ffmpeg_process = ffmpeg_process
            self.chunk_id = chunk_id

    def _play(self) -> None:
        if self.ffmpeg_process is None:
            print('video not loaded')
            return

        while not self.stop_event.is_set():
            in_bytes = self.ffmpeg_process.stdout.read(self.bytes_per_frame)
            if not in_bytes:
                # self.end_callback(self.chunk_id)
                break

            if self.frame_type == FrameType.VIDEO:
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([self.frame_height, self.frame_width, 3])
                frame = cvtColor(frame, COLOR_RGB2BGR)
            else:
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([self.frame_height, self.frame_width])

            self.frame_callback(self.cam_id, self.frame_type, frame)

        self.ffmpeg_process.stdout.close()

    def _get_video_dimensions(self, video_file: str) -> tuple:
        probe = ffmpeg.probe(video_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise Exception('No video stream found for file', video_file)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return width, height