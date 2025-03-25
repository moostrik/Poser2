import ffmpeg
import numpy as np
from threading import Thread, Event
from typing import Callable
from enum import Enum

from modules.cam.DepthAi.Definitions import FrameType, FrameTypeString

class DecoderType(Enum):
    CPU =   0
    GPU =   1
    iGPU =  2

DecoderString: dict[DecoderType, str] = {
    DecoderType.CPU:  'h264',
    DecoderType.GPU:  'h264_cuvid',
    DecoderType.iGPU: 'h264_qsv'
}

PlayerCallback = Callable[[int, FrameType, np.ndarray], None]
EndCallback = Callable[[int], None]

class Player:
    def __init__(self, cam_id: int, frameType: FrameType, frameCallback: PlayerCallback, endCallback: EndCallback, decoder: DecoderType) -> None:
        self.frame_callback: PlayerCallback = frameCallback
        self.end_callback: EndCallback = endCallback


        self.cam_id: int = cam_id
        self.frameType: FrameType = frameType
        self.vcodec: str = DecoderString[decoder]

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
        self.is_playing = False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def _play(self) -> None:
        try:
            width, height = self._get_video_dimensions(self.video_file)
        except Exception as e:
            self.is_playing = False
            return

        pix_fmt: str = 'rgb24' if self.frameType == FrameType.VIDEO else 'gray'

        process = (
            ffmpeg
            .input(self.video_file, re=None) # re: Read input at native frame rate. This is equivalent to setting -readrate 1.
            .output('pipe:', format='rawvideo', pix_fmt=pix_fmt, vcodec=self.vcodec)
            .run_async(pipe_stdout=True)
        )

        while self.is_playing and not self.stop_event.is_set():
            in_bytes = process.stdout.read(width * height * (3 if self.frameType == FrameType.VIDEO else 1))
            if not in_bytes:
                self.is_playing = False
                self.end_callback(self.chunk_id)
                break
            frame: np.ndarray = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            self.frame_callback(self.cam_id, self.frameType, frame)

        process.stdout.close()
        process.wait()

    def _get_video_dimensions(self, video_file: str) -> tuple:
        probe = ffmpeg.probe(video_file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise Exception('No video stream found for file', video_file)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        return width, height