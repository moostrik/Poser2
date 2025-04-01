
from threading import Thread, Lock, Event
from pathlib import Path
import time
from enum import Enum, auto
from queue import Queue, Empty

from modules.cam.recorder.FFmpegRecorder import FFmpegRecorder, EncoderType
from modules.cam.depthcam.Definitions import FrameType, FRAME_TYPE_LABEL_DICT

def make_path(path: Path, c: int, t: FrameType, chunk: int) -> Path:
    return path / f"{c}_{FRAME_TYPE_LABEL_DICT[t]}_{chunk:03d}.mp4"

class RecState(Enum):
    IDLE =  auto()
    START = auto()
    SPLIT = auto()
    STOP =  auto()

class SyncRecorder(Thread):
    def __init__(self, output_path: str, num_cams: int, types: list[FrameType], chunk_duration: float, encoder: EncoderType) -> None:
        super().__init__()
        self.output_path: Path = Path(output_path)
        self.num_cams: int = num_cams
        self.types: list[FrameType] = types
        self.recorders: dict[int, dict[FrameType, FFmpegRecorder]] = {}
        self.path: Path = Path()

        if FrameType.NONE in self.types:
            self.types.remove(FrameType.NONE)
        if FrameType.STEREO in self.types:
            self.types.remove(FrameType.STEREO)

        for c in range(num_cams):
            self.recorders[c] = {}
            for t in types:
                self.recorders[c][t] = FFmpegRecorder(encoder)

        self.chunk_duration: float = chunk_duration
        self.start_time: float
        self.chunk_index = 0
        self.rec_name: str
        self.fps: float = 30.0

        self.state: RecState = RecState.IDLE
        self.state_lock = Lock()

        self.settings_lock = Lock()

        self.stop_event = Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        self._set_state(RecState.IDLE)

        while not self.stop_event.is_set():

            if self._get_state() == RecState.STOP:
                self._stop_recording()
                self._set_state(RecState.IDLE)
            if self._get_state() == RecState.SPLIT:
                self._update_recording()
            if self._get_state() == RecState.START:
                self._start_recording()
                self._set_state(RecState.SPLIT)

            time.sleep(0.01)

        self._stop_recording()

    def _start_recording(self) -> None:
        self.rec_name = time.strftime("%Y%m%d-%H%M%S") + '_' + str(self.num_cams) + '_' + '_'.join([FRAME_TYPE_LABEL_DICT[t] for t in self.types])

        self.path = self.output_path / self.rec_name
        self.path.mkdir(parents=True, exist_ok=True)

        self.chunk_index = 0

        for c in range(self.num_cams):
            for t in self.types:
                path: Path = make_path(self.path, c, t, self.chunk_index)
                self.recorders[c][t].start(str(path), self.fps)

        self.start_time = time.time()
        self.recording = True

    def _stop_recording(self) -> None:
        for c in range(self.num_cams):
            for t in self.types:
                self.recorders[c][t].stop()

    def _update_recording(self) -> None:
        if time.time() - self.start_time > self.chunk_duration:
            self.chunk_index += 1
            fps: float = self.get_fps()

            for c in range(self.num_cams):
                for t in self.types:
                    path: Path = make_path(self.path, c, t, self.chunk_index)
                    self.recorders[c][t].split(str(path), fps)
            self.start_time += self.chunk_duration

    def _get_state(self) -> RecState:
        with self.state_lock:
            return self.state

    def _set_state(self, state: RecState) -> None:
        with self.state_lock:
            if state == RecState.START and self.state != RecState.IDLE:
                return
            if  state != RecState.IDLE and self.state == RecState.STOP:
                return

            self.state = state

    # EXTERNAL METHODS
    def add_frame(self, cam_id: int, t: FrameType, frame) -> None:
        self.recorders[cam_id][t].add_frame(frame)

    def set_fps(self, cam_id: int, fps: float) -> None:
        with self.settings_lock:
            self.fps = fps

    def get_fps(self) -> float:
        with self.settings_lock:
            return self.fps

    def record(self, value: bool) -> None:
        if value:
            self._set_state(RecState.START)
        else:
            self._set_state(RecState.STOP)

