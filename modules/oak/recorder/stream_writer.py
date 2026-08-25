from threading import Thread
import ffmpeg
import numpy as np
from queue import Queue, Empty
from enum import Enum
from time import sleep


class StreamWriter:
    def __init__(self, encoder_string: str, keyframe_interval: int = 0) -> None:
        self.output_file: str = ''
        self.fps: float = 30.0
        self.is_recording = False
        self.is_receiving = False
        self.frames = Queue()
        self.thread = None
        self.vcodec: str = encoder_string
        self.keyframe_interval: int = keyframe_interval

    def start(self, output_file: str, fps: float) -> None:
        if self.is_recording:
            return
        self.fps = fps
        self.output_file = output_file
        self.is_receiving = True
        self.thread = Thread(target=self._record)
        self.thread.start()

    def stop(self) -> None:
        if not self.is_recording:
            return
        self.is_receiving = False
        while not self.frames.empty():
            sleep(0.01)

        self.is_recording = False
        if self.thread is not None:
            self.thread.join()

    def split(self, output_file: str, fps: float) -> None:
        if not self.is_recording:
            return
        if self.is_recording:
            self.is_recording = False
        if self.thread is not None:
            self.thread.join()

        self.start(output_file, fps)

    def submit_frame(self, frame) -> None:
        if self.is_receiving:
            self.frames.put(frame)

    def _record(self) -> None:
        self.is_recording = True
        process = None

        while self.is_recording:
            frame: np.ndarray
            try:
                frame = self.frames.get(timeout=.1)
            except Empty:
                continue

            if process is None:
                width: int = frame.shape[1]
                height: int = frame.shape[0]
                pix_fmt: str = 'gray' if len(frame.shape) == 2 else 'bgr24'
                output_kwargs = dict(pix_fmt='yuv420p', vcodec=self.vcodec, r=self.fps)
                if self.keyframe_interval > 0:
                    output_kwargs['g'] = self.keyframe_interval          # GOP size
                    output_kwargs['keyint_min'] = self.keyframe_interval  # force exact spacing (no early IDR skip)
                process = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{width}x{height}', r=self.fps)
                    .output(self.output_file, **output_kwargs)
                    .overwrite_output()
                    .global_args('-loglevel', 'quiet')
                    .run_async(pipe_stdin=True)
                )

            if process is not None:
                process.stdin.write(frame.astype(np.uint8).tobytes())

        if process is not None:
            process.stdin.close()
            process.wait()