
from threading import Thread, Lock, Event
from pathlib import Path
import time
from enum import Enum, auto

from modules.Settings import Settings
from modules.cam.recorder.FFmpegRecorder import FFmpegRecorder
from modules.cam.depthcam.Definitions import FrameType, FRAME_TYPE_LABEL_DICT

def make_file_name(c: int, t: FrameType, chunk: int) -> str:
    return f"{c}_{FRAME_TYPE_LABEL_DICT[t]}_{chunk:03d}.mp4"

def make_folder_name(num_cams: int, color: bool, stereo: bool, lowres: bool) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + '_' + str(num_cams) + ('_color' if color else '_mono') + ('_stereo' if stereo else '') + ('_lowres' if lowres else '_highres')

def get_folder_name_settings(name: str) -> tuple[int, bool, bool, bool]:
    parts = name.split('_')
    if not parts[1].isdigit() or not ('color' in parts or 'mono' in parts) or not  ('lowres' in parts or 'highres' in parts):
        raise ValueError(f"Invalid folder name: {name}. Expected format: YYYYMMDD-HHMMSS_num_cams_color|mono_stereo|none_lowres|highres")
    num_cams = int(parts[1])
    color: bool = 'color' in parts
    stereo: bool = 'stereo' in parts
    lowres: bool = 'lowres' in parts
    return num_cams, color, stereo, lowres

EncoderString: dict[Settings.CoderType, str] = {
    Settings.CoderType.CPU:  'libx264',
    Settings.CoderType.GPU:  'h264_nvenc',
    Settings.CoderType.iGPU: 'h264_qsv'
}

class RecState(Enum):
    IDLE =  auto()
    START = auto()
    SPLIT = auto()
    STOP =  auto()

class SyncRecorder(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.output_path: Path = Path(settings.video_path)
        self.temp_path: Path = Path(settings.temp_path)

        self.settings: Settings = settings

        self.recorders: dict[int, dict[FrameType, FFmpegRecorder]] = {}
        self.fps: dict[int, float] = {}
        self.folder_path: Path = Path()

        for c in range(settings.num_cams):
            self.recorders[c] = {}
            self.fps[c] = settings.fps
            for t in self.settings.frame_types:
                self.recorders[c][t] = FFmpegRecorder(EncoderString[settings.encoder])

        self.start_time: float
        self.chunk_index = 0
        self.rec_name: str

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

        self.folder_path = self.output_path / make_folder_name(self.settings.num_cams, self.settings.color, self.settings.stereo, self.settings.lowres)
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.chunk_index = 0

        for c in range(self.settings.num_cams):
            fps: float = self.get_fps(c)
            for t in self.settings.frame_types:
                path: Path = self.folder_path / make_file_name(c, t, self.chunk_index)
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
                    path: Path = self.folder_path / make_file_name(c, t, self.chunk_index)
                    self.recorders[c][t].split(str(path), fps)
            self.start_time += self.settings.chunk_length

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
            self.fps[cam_id] = fps

    def get_fps(self, cam_id) -> float:
        with self.settings_lock:
            return self.fps[cam_id]

    def record(self, value: bool) -> None:
        if value:
            self._set_state(RecState.START)
        else:
            self._set_state(RecState.STOP)

