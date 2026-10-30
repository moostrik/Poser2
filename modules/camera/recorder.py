"""Records every camera's frames to chunked clips: one ffmpeg stream per camera and frame kind,
files named `<cam>_<kind>_<chunk>` in a folder named after the start time, written to a temp folder
and moved into place chunk by chunk."""

from threading import Thread, Lock, Event
from pathlib import Path
import shutil
import time
from enum import Enum, auto
from queue import Queue, Empty

from modules.settings import BaseSettings, Field, Widget
from modules.video import CoderType, CoderFormat, encoder_name, StreamWriter

from .frame import FrameType, CameraFrame, mirrored_image

import logging
logger = logging.getLogger(__name__)


class RecorderSettings(BaseSettings):

    # ── Initial settings ───────────────────────────────────────────────
    num_cams:        Field[int]              = Field(1, access=Field.INIT, description="Number of cameras")
    fps:                Field[float]            = Field(30.0, access=Field.INIT, description="Camera frame rate")
    keyframe_interval:  Field[int]              = Field(0, access=Field.INIT, description="Keyframe interval in frames (0 = encoder default, 1 = every frame)")
    temp_path:          Field[str]              = Field("temp", access=Field.INIT, description="Temporary files directory")
    video_encoder:      Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video encoder type")
    video_format:       Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, description="Video format")
    video_frame_types:  Field[list[FrameType]]  = Field([FrameType.VIDEO], access=Field.INIT, description="Frame types to record", visible=False)

    # ── Recording controls ────────────────────────────────────────────
    enabled:            Field[bool]             = Field(True, widget=Widget.switch, description="Enable video recording")
    start:              Field[bool]             = Field(False, widget=Widget.button, description="Start recording")
    stop:               Field[bool]             = Field(False, widget=Widget.button, description="Stop recording")
    recording:          Field[bool]             = Field(False, access=Field.READ, description="Recording")
    split:              Field[bool]             = Field(False, widget=Widget.button, description="Split chunk")
    output_path:        Field[str]              = Field("recordings", access=Field.INIT, description="Video recordings directory")
    name:               Field[str]              = Field("", widget=Widget.input, description="Recording name")


# The letter in a clip's file name: `0_C_001.mp4` is camera 0's video, chunk 1.
FRAME_TYPE_LABELS: dict[FrameType, str] = {
    FrameType.VIDEO: 'C',
}


def make_file_name(cam_id: int, frame_type: FrameType, chunk: int, suffix: str) -> str:
    return f"{cam_id}_{FRAME_TYPE_LABELS[frame_type]}_{chunk:03d}{suffix}"


def make_folder_name(name: str = "") -> str:
    folder = time.strftime("%Y%m%d-%H%M%S")
    if name:
        folder += f"_{name}"
    return folder


class RecState(Enum):
    IDLE =  auto()
    START = auto()
    REC =   auto()
    STOP =  auto()

class Recorder(Thread):
    def __init__(self, settings: RecorderSettings, data_path: str = "") -> None:
        super().__init__()
        self.temp_path: Path = Path(settings.temp_path)
        self.data_path: str = data_path

        self.settings: RecorderSettings = settings

        self.recorders: dict[int, dict[FrameType, StreamWriter]] = {}
        self.frames: dict[int, Queue[CameraFrame]] = {}
        self.folder_path: Path | None = None

        for c in range(settings.num_cams):
            self.recorders[c] = {}
            self.frames[c] = Queue()
            for t in self.settings.video_frame_types:
                self.recorders[c][t] = StreamWriter(encoder_name(settings.video_format, settings.video_encoder), settings.keyframe_interval)

        self.chunk_index = 0
        self.suffix: str = settings.video_format.value

        self.state: RecState = RecState.IDLE
        self.state_lock = Lock()
        self.stop_event = Event()

        self.settings.bind(RecorderSettings.start, self._on_start)
        self.settings.bind(RecorderSettings.stop, self._on_stop)
        self.settings.bind(RecorderSettings.split, self._on_split)
        self.settings.bind(RecorderSettings.enabled, self._on_enabled)

    def stop(self) -> None:
        self.stop_event.set()
        self.join()

    def run(self) -> None:
        self._set_state(RecState.IDLE)

        while not self.stop_event.is_set():
            try:
                if self._get_state() == RecState.STOP:
                    self._stop_recording()
                    self._set_state(RecState.IDLE)
                if self._get_state() == RecState.REC:
                    self._update_recording()
                if self._get_state() == RecState.START:
                    self._start_recording()
                    self._set_state(RecState.REC)
            except Exception:
                logger.exception("Recorder error")

            time.sleep(0.01)

        self._stop_recording()

    def _start_recording(self) -> None:

        self.folder_name = make_folder_name(self.settings.name)
        self.folder_path = self.temp_path / self.folder_name
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.output_folder_path = Path(self.data_path) / self.settings.output_path / self.folder_name
        self.output_folder_path.mkdir(parents=True, exist_ok=True)

        self.chunk_index = 0

        for c in range(self.settings.num_cams):
            for t in self.settings.video_frame_types:
                path: Path = self.folder_path / make_file_name(c, t, self.chunk_index, self.suffix)
                self.recorders[c][t].start(str(path), self.settings.fps)

        logger.info("VideoRecorder started → %s", self.output_folder_path)

    def _stop_recording(self) -> None:
        for c in range(self.settings.num_cams):
            for t in self.settings.video_frame_types:
                self.recorders[c][t].stop()

        self._move_chunk_files(self.chunk_index)
        self._remove_temp_folder()
        logger.info("VideoRecorder stopped")

    def _do_split(self) -> None:
        assert self.folder_path is not None
        old_chunk = self.chunk_index
        self.chunk_index += 1
        for c in range(self.settings.num_cams):
            for t in self.settings.video_frame_types:
                path: Path = self.folder_path / make_file_name(c, t, self.chunk_index, self.suffix)
                self.recorders[c][t].split(str(path), self.settings.fps)
        self._move_chunk_files(old_chunk)

    def _move_chunk_files(self, chunk_index: int) -> None:
        if not hasattr(self, 'output_folder_path'):
            return
        assert self.folder_path is not None     # set before output_folder_path
        for c in range(self.settings.num_cams):
            for t in self.settings.video_frame_types:
                name = make_file_name(c, t, chunk_index, self.suffix)
                src = self.folder_path / name
                dst = self.output_folder_path / name
                if src.exists():
                    try:
                        shutil.move(str(src), str(dst))
                    except Exception:
                        logger.exception("Failed to move %s → %s", src, dst)

    def _remove_temp_folder(self) -> None:
        if self.folder_path is not None and self.folder_path.exists():
            try:
                self.folder_path.rmdir()
            except OSError:
                logger.warning("Temp folder not empty, leaving: %s", self.folder_path)

    def _update_recording(self) -> None:
        for c in range(self.settings.num_cams):
            try:
                frame: CameraFrame = self.frames[c].get(timeout=0.01)
            except Empty:
                continue
            self.recorders[c][frame.kind].submit_frame(mirrored_image(frame) if frame.flip_h else frame.image, frame.nv12)

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
    def submit_frame(self, frame: CameraFrame) -> None:
        """A camera's frame callback: queued while recording, of the kinds the recorder writes."""
        if frame.kind in self.settings.video_frame_types and self._get_state() == RecState.REC:
            self.frames[frame.cam_id].put(frame)

    # SETTINGS CALLBACKS
    def _on_enabled(self, value: bool) -> None:
        if not value:
            self._on_stop()

    def _on_start(self, _=None) -> None:
        if self.settings.enabled and self._get_state() == RecState.IDLE:
            self._set_state(RecState.START)
            self.settings.recording = True

    def _on_stop(self, _=None) -> None:
        if self._get_state() in (RecState.START, RecState.REC):
            self._set_state(RecState.STOP)
            self.settings.recording = False

    def _on_split(self, _=None) -> None:
        if self._get_state() == RecState.REC:
            self._do_split()
