"""Records every camera's frames to chunked clips: one ffmpeg stream per camera and frame kind,
files named `<cam>_<kind>_<chunk>` in a folder named after the start time, written to a temp folder
and moved into place chunk by chunk.

With DEPTH in `video_frame_types` a chunk ends as one `<cam>_<chunk>.mkv` (`chunk.merge_chunk`): each
colour frame is written with the newest depth, the pairing the app shows live, so the two streams
count alike, and a metadata line per frame goes beside them. Before the first depth arrives nothing
is written. Recordings hold what the camera delivered, mirrored and not straightened, so the
correction can still be tuned on them. Both streams share `keyframe_interval`: a seek in a chunk goes
to a colour keyframe, so a depth stream with its keyframes elsewhere comes back with its references missing."""

from threading import Thread, Lock, Event
from pathlib import Path
import shutil
import time
from enum import Enum, auto
from queue import Queue, Empty
from typing import TextIO

from modules.settings import BaseSettings, Field, Widget
from modules.video import CoderType, CoderFormat, DEPTH_ENCODER, encoder_name, StreamWriter

from .chunk import METADATA_SUFFIX, ChunkInfo, finish_unmerged, info_line, merge_chunk, readings_line
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
    video_quality:      Field[int]              = Field(0, min=0, max=51, access=Field.INIT, description="Video constant quality, lower is better (0 = encoder default)")
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
    FrameType.DEPTH: 'D',
}
METADATA_LABEL: str = 'M'
DEPTH_SUFFIX: str = '.mkv'


def make_file_name(cam_id: int, frame_type: FrameType, chunk: int, suffix: str) -> str:
    return f"{cam_id}_{FRAME_TYPE_LABELS[frame_type]}_{chunk:03d}{suffix}"


def make_chunk_name(cam_id: int, chunk: int) -> str:
    """The one file a chunk with depth ends as."""
    return f"{cam_id}_{chunk:03d}.mkv"


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

        self.with_depth: bool = FrameType.DEPTH in settings.video_frame_types
        self.recorders: dict[int, dict[FrameType, StreamWriter]] = {}
        self.frames: dict[int, Queue[tuple[CameraFrame, CameraFrame | None]]] = {}    # a colour frame and its depth
        self.newest_depth: dict[int, CameraFrame | None] = {}
        self.metadata: dict[int, TextIO] = {}
        self.metadata_frames: dict[int, int] = {}       # the frames of the chunk so far, per camera
        self.folder_path: Path | None = None

        for c in range(settings.num_cams):
            self.recorders[c] = {}
            self.frames[c] = Queue()
            self.newest_depth[c] = None
            for t in self.settings.video_frame_types:
                vcodec: str = DEPTH_ENCODER if t == FrameType.DEPTH else encoder_name(settings.video_format, settings.video_encoder)
                self.recorders[c][t] = StreamWriter(vcodec, settings.keyframe_interval, settings.video_quality)

        if FrameType.DEPTH in settings.video_frame_types and settings.keyframe_interval <= 0:
            logger.warning("keyframe_interval 0 with depth: each encoder picks its own keyframes, so seeking will show the wrong depth")

        self.chunk_index = 0
        self.suffix: str = settings.video_format.value
        self.split_requested = Event()
        self.merges: list[Thread] = []

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

    def _stream_path(self, cam_id: int, frame_type: FrameType, chunk_index: int) -> Path:
        assert self.folder_path is not None
        suffix: str = DEPTH_SUFFIX if frame_type == FrameType.DEPTH else self.suffix
        return self.folder_path / make_file_name(cam_id, frame_type, chunk_index, suffix)

    def _metadata_path(self, cam_id: int, chunk_index: int) -> Path:
        assert self.folder_path is not None
        return self.folder_path / f"{cam_id}_{METADATA_LABEL}_{chunk_index:03d}{METADATA_SUFFIX}"

    def _open_metadata(self) -> None:
        for c in range(self.settings.num_cams):
            self.metadata[c] = self._metadata_path(c, self.chunk_index).open('w', encoding='utf-8')
            self.metadata_frames[c] = 0

    def _close_metadata(self) -> None:
        for file in self.metadata.values():
            file.close()
        self.metadata.clear()

    def _start_recording(self) -> None:
        output_path: Path = Path(self.data_path) / self.settings.output_path
        if self.with_depth:
            finish_unmerged(self.temp_path, output_path)

        self.folder_name = make_folder_name(self.settings.name)
        self.folder_path = self.temp_path / self.folder_name
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.output_folder_path = output_path / self.folder_name
        self.output_folder_path.mkdir(parents=True, exist_ok=True)

        self.chunk_index = 0

        for c in range(self.settings.num_cams):
            self.newest_depth[c] = None
            for t in self.settings.video_frame_types:
                self.recorders[c][t].start(str(self._stream_path(c, t, self.chunk_index)), self.settings.fps)
        if self.with_depth:
            self._open_metadata()

        logger.info("VideoRecorder started → %s", self.output_folder_path)

    def _stop_recording(self) -> None:
        self._write_queued()
        for c in range(self.settings.num_cams):
            for t in self.settings.video_frame_types:
                self.recorders[c][t].stop()

        self._close_metadata()
        self._finish_chunk(self.chunk_index)
        for merge in self.merges:
            merge.join()
        self.merges.clear()
        self._remove_temp_folder()
        logger.info("VideoRecorder stopped")

    def _do_split(self) -> None:
        """On the recorder's thread, between frames, so every stream of a chunk holds the same frames."""
        assert self.folder_path is not None
        old_chunk = self.chunk_index
        self.chunk_index += 1
        for c in range(self.settings.num_cams):
            for t in self.settings.video_frame_types:
                self.recorders[c][t].split(str(self._stream_path(c, t, self.chunk_index)), self.settings.fps)
        if self.with_depth:
            self._close_metadata()
            self._open_metadata()
        self._finish_chunk(old_chunk)

    def _finish_chunk(self, chunk_index: int) -> None:
        if not self.with_depth:
            self._move_chunk_files(chunk_index)
            return
        if self.folder_path is None:
            return
        for c in range(self.settings.num_cams):
            merge = Thread(target=self._merge_chunk, args=(c, chunk_index))     # the next chunk must not wait for it
            merge.start()
            self.merges.append(merge)

    def _merge_chunk(self, cam_id: int, chunk_index: int) -> None:
        video: Path = self._stream_path(cam_id, FrameType.VIDEO, chunk_index)
        depth: Path = self._stream_path(cam_id, FrameType.DEPTH, chunk_index)
        metadata: Path = self._metadata_path(cam_id, chunk_index)
        if not (video.exists() and depth.exists() and metadata.exists()):
            for part in (video, depth, metadata):
                part.unlink(missing_ok=True)    # a chunk that never got a frame
            return
        merge_chunk(video, depth, metadata, self.output_folder_path / make_chunk_name(cam_id, chunk_index))

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
        if self.split_requested.is_set():
            self.split_requested.clear()
            self._do_split()
        self._write_queued()

    def _write_queued(self) -> None:
        """Everything queued, on every pass: with depth two streams arrive per camera at the full
        frame rate, more than one frame per pass can keep up with."""
        for c in range(self.settings.num_cams):
            while True:
                try:
                    frame, depth = self.frames[c].get_nowait()
                except Empty:
                    break
                self.recorders[c][frame.kind].submit_frame(mirrored_image(frame) if frame.flip_h else frame.image, frame.nv12)
                if depth is not None:
                    self.recorders[c][depth.kind].submit_frame(mirrored_image(depth) if depth.flip_h else depth.image)
                    self._write_metadata(c, frame)

    def _write_metadata(self, cam_id: int, frame: CameraFrame) -> None:
        file: TextIO | None = self.metadata.get(cam_id)
        if file is None:
            return
        count: int = self.metadata_frames[cam_id]
        if count == 0:
            height: int = frame.image.shape[0] * 2 // 3 if frame.nv12 else frame.image.shape[0]
            file.write(info_line(ChunkInfo(frame.image.shape[1], height, frame.intrinsics, frame.flip_h, self.settings.fps)))
        if frame.readings is not None:
            file.write(readings_line(count, frame.timestamp, frame.readings))
        self.metadata_frames[cam_id] = count + 1

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
        """A camera's frame callback: queued while recording, of the kinds the recorder writes. A
        depth frame is only held; each colour frame takes the newest along."""
        if frame.kind not in self.settings.video_frame_types or self._get_state() != RecState.REC:
            return
        if frame.kind == FrameType.DEPTH:
            self.newest_depth[frame.cam_id] = frame
            return
        depth: CameraFrame | None = self.newest_depth[frame.cam_id]
        if self.with_depth and depth is None:
            return
        self.frames[frame.cam_id].put((frame, depth))

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
            self.split_requested.set()
