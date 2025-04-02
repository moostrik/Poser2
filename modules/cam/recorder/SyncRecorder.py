
from threading import Thread, Lock, Event
from pathlib import Path
import time
from enum import Enum, auto

from modules.Settings import Settings
from modules.cam.recorder.FFmpegRecorder import FFmpegRecorder
from modules.cam.depthcam.Definitions import FrameType, FRAME_TYPE_LABEL_DICT

def make_file_name(c: int, t: FrameType, chunk: int) -> str:
    return f"{c}_{FRAME_TYPE_LABEL_DICT[t]}_{chunk:03d}.mp4"

def make_folder_name(num_cams: int, color: bool, lowres: bool) -> str:
    return time.strftime("%Y%m%d-%H%M%S") + '_' + str(num_cams) + ('_color' if color else '_mono') + ('_lowres' if lowres else '_highres')

def get_folder_name_settings(name: str) -> tuple[int, bool, bool]:
    parts = name.split('_')
    if len(parts) != 4 or not parts[1].isdigit() or not parts[2] in ['color', 'mono'] or not parts[3] in ['lowres', 'highres']:
        raise ValueError(f"Invalid folder name: {name}. Expected format: YYYYMMDD-HHMMSS_num_cams_mono|stereo_lowres|highres")
    num_cams = int(parts[1])
    color: bool = parts[2] == 'color'
    lowres: bool = parts[3] == 'lowres'
    return num_cams, color, lowres

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
    def __init__(self, settings: Settings, num_cams: int, types: list[FrameType]) -> None:
        super().__init__()
        self.output_path: Path = Path(settings.video_path)
        self.temp_path: Path = Path(settings.temp_path)

        self.num_cams: int = num_cams
        self.color: bool = settings.color
        self.lowres: bool = settings.lowres

        self.types: list[FrameType] = types
        self.recorders: dict[int, dict[FrameType, FFmpegRecorder]] = {}
        self.folder_path: Path = Path()

        if FrameType.NONE in self.types:
            self.types.remove(FrameType.NONE)
        if FrameType.STEREO in self.types:
            self.types.remove(FrameType.STEREO)

        for c in range(num_cams):
            self.recorders[c] = {}
            for t in types:
                self.recorders[c][t] = FFmpegRecorder(EncoderString[settings.encoder])

        self.chunk_length: float = settings.chunk_length
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

        self.folder_path = self.output_path / make_folder_name(self.num_cams, self.color, self.lowres)
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.chunk_index = 0

        for c in range(self.num_cams):
            for t in self.types:
                path: Path = self.folder_path / make_file_name(c, t, self.chunk_index)
                self.recorders[c][t].start(str(path), self.fps)

        self.start_time = time.time()
        self.recording = True

    def _stop_recording(self) -> None:
        for c in range(self.num_cams):
            for t in self.types:
                self.recorders[c][t].stop()

    def _update_recording(self) -> None:
        if time.time() - self.start_time > self.chunk_length:
            self.chunk_index += 1
            fps: float = self.get_fps()

            for c in range(self.num_cams):
                for t in self.types:
                    path: Path = self.folder_path / make_file_name(c, t, self.chunk_index)
                    self.recorders[c][t].split(str(path), fps)
            self.start_time += self.chunk_length

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

