import ffmpeg
import numpy as np
from threading import Thread, Event
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

class Player:
    def __init__(self, cam_id: int, frameType: FrameType, frameCallback: PlayerCallback, endCallback: EndCallback, hw_acceleration: HwAccelerationType) -> None:
        self.frame_callback: PlayerCallback = frameCallback
        self.end_callback: EndCallback = endCallback

        self.cam_id: int = cam_id
        self.frameType: FrameType = frameType
        self.hw_acceleration: HwAccelerationType = hw_acceleration

        self.is_playing = False
        self.thread = None
        self.stop_event = Event()

        self.video_file: str
        self.chunk_id: int

    def start(self, video_file: str, chunk_id: int) -> None:
        if self.is_playing:
            print('Already playing')
            return

        self.is_playing = True
        self.thread = Thread(target=self._play)
        self.stop_event.clear()

        self.video_file = video_file
        self.chunk_id = chunk_id

        self.thread.start()

    def stop(self) -> None:
        # print('Stopping player', self.cam_id, self.frameType)
        self.is_playing = False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def _play(self) -> None:
        try:
            width, height = self._get_video_dimensions(self.video_file)
        except Exception as e:
            self.is_playing = False
            return

        pix_fmt: str = 'rgb24' if self.frameType == FrameType.VIDEO else 'gray'
        bytes_per_frame = width * height * (3 if self.frameType == FrameType.VIDEO else 1)

        T: float = time.time()
        process = (
            ffmpeg
            .input(self.video_file, re=None, hwaccel='d3d12va', hwaccel_device='1')  # Use Intel iGPU (device 1)
            .output('pipe:', format='rawvideo', pix_fmt=pix_fmt)
            .global_args('-loglevel', 'quiet')
            .run_async(pipe_stdout=True)
        )
        # print(T - time.time())
        p = True

        while self.is_playing and not self.stop_event.is_set():
            in_bytes = process.stdout.read(bytes_per_frame)
            if p:
                # print(T - time.time())
                p = False
            if not in_bytes:
                self.is_playing = False
                self.end_callback(self.chunk_id)
                break

            if pix_fmt == 'rgb24':
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
                frame = cvtColor(frame, COLOR_RGB2BGR)
            else:
                frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([height, width])

            self.frame_callback(self.cam_id, self.frameType, frame)

        # print('Closing', self.cam_id, self.frameType)
        process.stdout.close()
        # process.wait()
        # print('Closed', self.cam_id, self.frameType)

    def _get_video_dimensions(self, video_file: str) -> tuple:
        probe = ffmpeg.probe(video_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise Exception('No video stream found for file', video_file)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return width, height