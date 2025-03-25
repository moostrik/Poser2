import cv2
import threading
from collections import deque

class Recorder:
    def __init__(self) -> None:
        self.output_file: str = ''
        self.fps: int = 30
        self.frame_size: tuple[int, int] | None = None
        self.is_recording = False
        self.frames = deque()
        self.lock = threading.Lock()
        self.thread = None

    def start(self, output_file: str, fps: int) -> None:
        if self.is_recording:
            print('Already recording')
            return
        self.fps = fps
        self.output_file: str = output_file
        self.is_recording = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def stop(self) -> None:
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()

    def add_frame(self, frame) -> None:
        if self.is_recording:
            with self.lock:
                if self.frame_size is None:
                    # if the frame uses one channel convert it to 3 channels
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                    self.frame_size = (frame.shape[1], frame.shape[0])
                self.frames.append(frame)

    def _record(self):
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # type: ignore
        out = None

        while self.is_recording or self.frames:
            with self.lock:
                if self.frames:
                    frame = self.frames.popleft()
                        # Allocate VideoWriter after the first frame is received
                    if out is None and self.frame_size is not None:
                        out = cv2.VideoWriter(self.output_file, fourcc, self.fps, self.frame_size)
                    if out is not None:
                        out.write(frame)

        if out is not None:
            out.release()
