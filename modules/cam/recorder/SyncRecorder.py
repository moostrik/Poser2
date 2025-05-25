
from threading import Thread, Lock, Event
from pathlib import Path
import time
from enum import Enum, auto
from numpy import ndarray
from queue import Queue, Empty

from modules.Settings import Settings
from modules.cam.recorder.FFmpegRecorder import FFmpegRecorder
from modules.cam.depthcam.Definitions import FrameType, FRAME_TYPE_LABEL_DICT

def make_file_name(c: int, t: FrameType, chunk: int, format: str) -> str:
    return f"{c}_{FRAME_TYPE_LABEL_DICT[t]}_{chunk:03d}{format}"

def make_folder_name(num_cams: int, square: bool, color: bool, stereo: bool) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + '_' + str(num_cams) + ('_square' if square else '_wide') + ('_color' if color else '_mono') + ('_stereo' if stereo else '')

def is_folder_for_settings(name: str, settings: Settings) -> bool:
    parts: list[str] = name.split('_')
    if len(parts) < 2:
        return False
    if not parts[1].isdigit() or not  ('wide' in parts or 'square' in parts) or not ('color' in parts or 'mono' in parts):
        return False
    num_cams = int(parts[1])
    square: bool = 'square' in parts
    color: bool = 'color' in parts
    stereo: bool = 'stereo' in parts
    if num_cams >= settings.num_cams and square == settings.square and color == settings.color and stereo == settings.stereo:
        return True
    return False

EncoderString: dict[Settings.CoderFormat, dict[Settings.CoderType, str]] = {
    Settings.CoderFormat.H264: {
        Settings.CoderType.CPU:  'libx264',
        Settings.CoderType.GPU:  'h264_nvenc',
        Settings.CoderType.iGPU: 'h264_qsv'
    },
    Settings.CoderFormat.H265: {
        Settings.CoderType.CPU:  'libx265',
        Settings.CoderType.GPU:  'hevc_nvenc',
        Settings.CoderType.iGPU: 'hevc_qsv'
    }
}

class RecState(Enum):
    IDLE =  auto()
    START = auto()
    REC =   auto()
    STOP =  auto()

class SyncRecorder(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.output_path: Path = Path(settings.video_path)
        self.temp_path: Path = Path(settings.temp_path)

        self.settings: Settings = settings

        self.recorders: dict[int, dict[FrameType, FFmpegRecorder]] = {}
        self.fps: dict[int, float] = {}
        self.frames: dict[int, Queue[dict[FrameType, ndarray]]] = {}
        self.folder_path: Path = Path()

        for c in range(settings.num_cams):
            self.recorders[c] = {}
            self.fps[c] = settings.fps
            self.frames[c] = Queue()
            for t in self.settings.frame_types:
                self.recorders[c][t] = FFmpegRecorder(EncoderString[settings.format][settings.encoder])

        self.start_time: float
        self.chunk_index = 0
        self.rec_name: str
        self.suffix: str = settings.format.value

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
            if self._get_state() == RecState.REC:
                self._update_recording()
            if self._get_state() == RecState.START:
                self._start_recording()
                self._set_state(RecState.REC)

            time.sleep(0.01)

        self._stop_recording()

    def _start_recording(self) -> None:

        self.folder_path = self.output_path / make_folder_name(self.settings.num_cams, self.settings.square, self.settings.color, self.settings.stereo)
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.chunk_index = 0

        for c in range(self.settings.num_cams):
            fps: float = self.get_fps(c)
            for t in self.settings.frame_types:
                path: Path = self.folder_path / make_file_name(c, t, self.chunk_index, self.suffix)
                self.recorders[c][t].start(str(path), fps)

        self.start_time = time.time()
        self.recording = True

    def _stop_recording(self) -> None:
        for c in range(self.settings.num_cams):
            for t in self.settings.frame_types:
                self.recorders[c][t].stop()

    def _update_recording(self) -> None:
        if time.time() - self.start_time > self.settings.chunk_length:
            self.chunk_index += 1

            for c in range(self.settings.num_cams):
                fps: float = self.get_fps(c)
                for t in self.settings.frame_types:
                    path: Path = self.folder_path / make_file_name(c, t, self.chunk_index, self.suffix)
                    self.recorders[c][t].split(str(path), fps)
            self.start_time += self.settings.chunk_length

        for c in range(self.settings.num_cams):
            try:
                frames: dict[FrameType, ndarray] = self.frames[c].get(timeout=0.01)
            except Empty:
                continue
            for t in self.settings.frame_types:
                if t in frames:
                    self.recorders[c][t].add_frame(frames[t])

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
    def add_synced_frames(self, cam_id: int, frames: dict[FrameType, ndarray], fps: float) -> None:
        self.set_fps(cam_id, fps)
        if self._get_state() == RecState.REC:
            self.frames[cam_id].put(frames)

    def set_fps(self, cam_id: int, fps: float) -> None:
        with self.settings_lock:
            self.fps[cam_id] = fps

    def get_fps(self, cam_id) -> float:
        with self.settings_lock:
            return self.fps[cam_id]

    def record(self, value: bool) -> None:
        if value:
            self._set_state(RecState.START)
        else:
            self._set_state(RecState.STOP)

