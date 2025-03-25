from threading import Thread
import ffmpeg
import numpy as np
from queue import Queue, Empty
from enum import Enum

class EncoderType(Enum):
    CPU =   0
    GPU =   1
    iGPU =  2

EncoderString: dict[EncoderType, str] = {
    EncoderType.CPU:  'libx264',
    EncoderType.GPU:  'h264_nvenc',
    EncoderType.iGPU: 'h264_qsv'
}

class Recorder:
    def __init__(self, encoder: EncoderType) -> None:
        self.output_file: str = ''
        self.fps: float = 30.0
        self.is_recording = False
        self.frames = Queue()
        self.thread = None
        self.vcodec: str = EncoderString[encoder]


    def start(self, output_file: str, fps: float) -> None:
        if self.is_recording:
            print('Already recording')
            return
        self.fps = fps
        self.output_file = output_file
        self.is_recording = True
        self.thread = Thread(target=self._record)
        self.thread.start()

    def stop(self) -> None:
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()

    def add_frame(self, frame) -> None:
        if self.is_recording:
            self.frames.put(frame)

    def _record(self) -> None:
        process = None

        while self.is_recording or not self.frames.empty():
            frame: np.ndarray
            try:
                frame = self.frames.get(timeout=.1)
            except Empty:
                continue

            if process is None:
                width: int = frame.shape[1]
                height: int = frame.shape[0]
                pix_fmt: str = 'gray' if len(frame.shape) == 2 else 'bgr24'
                process = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}', r=self.fps)
                    .output(self.output_file, pix_fmt='yuv420p', vcodec=self.vcodec, r=self.fps)
                    .overwrite_output()
                    .global_args('-loglevel', 'quiet')
                    .run_async(pipe_stdin=True)
                )

            if process is not None:
                process.stdin.write(frame.astype(np.uint8).tobytes())

        if process is not None:
            process.stdin.close()
            process.wait()