"""One ffmpeg encoder process fed raw frames from a queue: the writer of one camera recording stream."""

from threading import Thread
import ffmpeg
import numpy as np
from queue import Queue, Empty
from time import sleep


class StreamWriter:
    def __init__(self, vcodec: str, keyframe_interval: int = 0) -> None:
        self.output_file: str = ''
        self.fps: float = 30.0
        self.is_recording = False
        self.is_receiving = False
        self.frames = Queue()
        self.thread = None
        self.vcodec: str = vcodec
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

    def submit_frame(self, frame: np.ndarray, nv12: bool = False) -> None:
        """Queue one frame: BGR (H, W, 3), gray (H, W), or with `nv12` packed NV12 (H * 3 / 2, W),
        which needs saying because it is 2-D too."""
        if self.is_receiving:
            self.frames.put((frame, nv12))

    def _record(self) -> None:
        self.is_recording = True
        process = None

        while self.is_recording:
            frame: np.ndarray
            nv12: bool
            try:
                frame, nv12 = self.frames.get(timeout=.1)
            except Empty:
                continue

            if process is None:
                width: int = frame.shape[1]
                height: int = frame.shape[0] * 2 // 3 if nv12 else frame.shape[0]
                pix_fmt: str = 'nv12' if nv12 else 'gray' if len(frame.shape) == 2 else 'bgr24'
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
