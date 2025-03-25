from modules.cam.recorder.Recorder import Recorder
import time
import os
from threading import Thread, Event

from modules.cam.DepthAi.Definitions import FrameType, FrameTypeNames

class SyncRecorder(Thread):
    def __init__(self, output_path: str, types: list[FrameType], chunk_duration: float) -> None:
        super().__init__()
        self.output_path: str = output_path
        self.types: list[FrameType] = types
        self.recorders: dict[FrameType, Recorder] = {}
        self.paths: dict[FrameType, str] = {}

        if FrameType.NONE in self.types:
            self.types.remove(FrameType.NONE)
        if FrameType.STEREO in self.types:
            self.types.remove(FrameType.STEREO)

        # self.types: list[FrameType] = [FrameType.VIDEO]

        for t in types:
            self.recorders[t] = Recorder()
            self.paths[t] = ''

        self.chunk_duration: float = chunk_duration
        self.start_time: float
        self.chunk_index = 0
        self.rec_name: str
        self.fps: float = 30.0

        self.start_recording_event = Event()
        self.stop_recording_event = Event()
        self.stop_event = Event()
        self.recording = False
        self.running = False

    def run(self) -> None:
        self.running = True

        while self.running:
            if self.stop_event.is_set():
                self.stop_recording_event.set()
                self.running = False
            if self.start_recording_event.is_set():
                self.start_recording_event.clear()
                self._start_recording()
            if self.stop_recording_event.is_set():
                self.stop_recording_event.clear()
                self._stop_recording()
            self._update_recording()
            time.sleep(0.01)

    def stop(self) -> None:
        self.stop_event.set()
        self.join()

    def _start_recording(self) -> None:
        if self.recording:
            return
        print('Start recording')

        self.rec_name = time.strftime("%Y%m%d-%H%M%S")
        path: str = self.output_path + '/' + self.rec_name

        self.chunk_index = 0
        chunk_name: str =  f"{self.chunk_index:04d}"

        for t in self.types:
            self.paths[t] = path + '/' + t.name + '/'
            os.makedirs(self.paths[t] , exist_ok=False)
            full_path: str = self.paths[t] + chunk_name + '.mp4'
            print(full_path)
            self.recorders[t].start(full_path, self.fps)

        self.start_time = time.time()
        self.recording = True

    def _stop_recording(self) -> None:
        if not self.recording:
            return
        print('Stop recording')
        self.recording = False
        for t in self.types:
            self.recorders[t].stop()

    def _update_recording(self) -> None:
        if self.recording:
            if time.time() - self.start_time > self.chunk_duration:
                self.chunk_index += 1
                chunk_name: str = f"{self.chunk_index:04d}"

                for t in self.types:
                    self.recorders[t].stop()
                    full_path: str = self.paths[t] + chunk_name + '.mp4'
                    self.recorders[t].start(full_path, 30)
                self.start_time += self.chunk_duration

    # EXTERNAL METHODS
    def add_frame(self, cam_id: int, t: FrameType, frame) -> None:
        self.recorders[t].add_frame(frame)

    def set_fps(self, cam_id: int, fps: float) -> None:
        self.fps = fps

    def start_recording(self) -> None:
        print('Start')
        self.start_recording_event.set()

    def stop_recording(self) -> None:
        print('Stop')
        self.stop_recording_event.set()

