from threading import Thread, Lock, Event
from pathlib import Path
import shutil
import time
from enum import Enum, auto
from numpy import ndarray
from queue import Queue, Empty

from ..camera.definitions import CoderType, CoderFormat, FrameType, FRAME_TYPE_LABEL_DICT
from .settings import RecorderSettings
from .stream_writer import StreamWriter

import logging
logger = logging.getLogger(__name__)

def make_file_name(cam_id: int, frame_type: FrameType, chunk: int, suffix: str) -> str:
    return f"{cam_id}_{FRAME_TYPE_LABEL_DICT[frame_type]}_{chunk:03d}{suffix}"

def make_folder_name(name: str = "") -> str:
    folder = time.strftime("%Y%m%d-%H%M%S")
    if name:
        folder += f"_{name}"
    return folder

EncoderString: dict[CoderFormat, dict[CoderType, str]] = {
    CoderFormat.H264: {
        CoderType.CPU:  'libx264',
        CoderType.GPU:  'h264_nvenc',
        CoderType.iGPU: 'h264_qsv'
    },
    CoderFormat.H265: {
        CoderType.CPU:  'libx265',
        CoderType.GPU:  'hevc_nvenc',
        CoderType.iGPU: 'hevc_qsv'
    }
}

class RecState(Enum):
    IDLE =  auto()
    START = auto()
    REC =   auto()
    STOP =  auto()

class Recorder(Thread):
    def __init__(self, settings: RecorderSettings) -> None:
        super().__init__()
        self.temp_path: Path = Path(settings.temp_path)

        self.settings: RecorderSettings = settings

        self.recorders: dict[int, dict[FrameType, StreamWriter]] = {}
        self.fps: dict[int, float] = {}
        self.frames: dict[int, Queue[dict[FrameType, ndarray]]] = {}
        self.folder_path: Path = Path()

        for c in range(settings.num_cameras):
            self.recorders[c] = {}
            self.fps[c] = settings.fps
            self.frames[c] = Queue()
            for t in self.settings.video_frame_types:
                self.recorders[c][t] = StreamWriter(EncoderString[settings.video_format][settings.video_encoder])

        self.start_time: float = 0.0
        self.chunk_index = 0
        self.suffix: str = settings.video_format.value

        self.state: RecState = RecState.IDLE
        self.state_lock = Lock()
        self.settings_lock = Lock()
        self.stop_event = Event()

        self.settings.bind(RecorderSettings.record, self._on_record)
        self.settings.bind(RecorderSettings.split, self._on_split)

    def stop(self) -> None:
        self.stop_event.set()
        self.join()

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

        self.folder_name = make_folder_name(self.settings.name)
        self.folder_path = self.temp_path / self.folder_name
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.chunk_index = 0

        for c in range(self.settings.num_cameras):
            fps: float = self.get_fps(c)
            for t in self.settings.video_frame_types:
                path: Path = self.folder_path / make_file_name(c, t, self.chunk_index, self.suffix)
                self.recorders[c][t].start(str(path), fps)

        self.start_time = time.time()
        logger.info("VideoRecorder started → %s", self.folder_path)

    def _stop_recording(self) -> None:
        for c in range(self.settings.num_cameras):
            for t in self.settings.video_frame_types:
                self.recorders[c][t].stop()

        # Move completed recording from temp to final location
        if hasattr(self, 'folder_name') and self.folder_path.exists():
            output_path = Path(self.settings.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            destination = output_path / self.folder_name
            try:
                shutil.move(str(self.folder_path), str(destination))
            except Exception:
                logger.exception("Failed to move recording from %s to %s", self.folder_path, destination)

        logger.info("VideoRecorder stopped")

    def _do_split(self) -> None:
        self.chunk_index += 1
        for c in range(self.settings.num_cameras):
            fps: float = self.get_fps(c)
            for t in self.settings.video_frame_types:
                path: Path = self.folder_path / make_file_name(c, t, self.chunk_index, self.suffix)
                self.recorders[c][t].split(str(path), fps)
        self.start_time += self.settings.chunk_length

    def _update_recording(self) -> None:
        # Self-chunk when chunk_length > 0 (standalone mode)
        if self.settings.chunk_length > 0 and time.time() - self.start_time > self.settings.chunk_length:
            self._do_split()

        for c in range(self.settings.num_cameras):
            try:
                frames: dict[FrameType, ndarray] = self.frames[c].get(timeout=0.01)
            except Empty:
                continue
            for t in self.settings.video_frame_types:
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
    def set_synced_frames(self, cam_id: int, frames: dict[FrameType, ndarray], fps: float) -> None:
        self.set_fps(cam_id, fps)
        if self._get_state() == RecState.REC:
            self.frames[cam_id].put(frames)

    def set_fps(self, cam_id: int, fps: float) -> None:
        with self.settings_lock:
            self.fps[cam_id] = fps

    def get_fps(self, cam_id) -> float:
        with self.settings_lock:
            return self.fps[cam_id]

    # SETTINGS CALLBACKS
    def _on_record(self, value: bool) -> None:
        if value:
            self._set_state(RecState.START)
        else:
            self._set_state(RecState.STOP)

    def _on_split(self, _=None) -> None:
        if self._get_state() == RecState.REC:
            self._do_split()


