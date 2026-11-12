"""Records every camera's frames to chunked clips: one ffmpeg stream per camera and frame kind,
files named `<cam>_<kind>_<chunk>` in a folder named after the start time, written to a temp folder
and merged into one `<cam>_<chunk>.mkv` per camera when a chunk ends (`chunk.merge_chunk`), with the
metadata attached: the chunk's info line and one line per frame.

The parts of one capture are brought together by a `RecordingSync`: the color frame, its depth when
DEPTH is in `video_frame_types` and the camera sends it, the camera's tracklets and the host's poses,
each part settled by the order its stream comes in. A camera that sends no depth is recorded color
only. What waits for the encoders, queued here and in the writers, is bounded in bytes (`HELD`); beyond
it a capture is dropped whole, before any part of it is written, so every stream and the metadata of a
chunk hold the same frames. A split cuts every camera's chunk at one capture time: a camera goes on in
the new chunk with its first capture after it. On stop no frame is taken any more, and the captures
still waiting are settled by the parts that still come, for at most `STOP_WAIT`. Recordings hold what
the camera delivered, not straightened and not mirrored, so the correction can still be tuned on them;
`flip_h` says the frames are still to be mirrored. What a capture went without is counted on the panel
and warned of. Both streams share `keyframe_interval`: a seek in a chunk goes to a color keyframe, so a
depth stream with its keyframes elsewhere comes back with its references missing."""

from threading import Thread, Lock, Event
from pathlib import Path
import time
from enum import Enum, auto
from queue import Queue, Empty
from typing import TextIO

from modules.settings import BaseSettings, Field, Widget
from modules.video import CoderType, CoderFormat, DEPTH_ENCODER, encoder_name, StreamWriter

from .chunk import DEPTH_SUFFIX, METADATA_LABEL, METADATA_SUFFIX, ChunkInfo, FrameMetadata, finish_unmerged, frame_line, info_line, \
    make_chunk_name, make_file_name, make_folder_name, merge_chunk
from .frame import FrameType, CameraFrame
from .recording_sync import FramePoses, RecordedCapture, RecordingSync
from .tracklet import CameraTracklets

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

    # ── What a recording went without ─────────────────────────────────
    missed_depth:       Field[int]              = Field(0, access=Field.READ, description="Color frames left out, their depth missing")
    missed_color:       Field[int]              = Field(0, access=Field.READ, description="Depth frames left out, their color frame missing")
    missed_tracklets:   Field[int]              = Field(0, access=Field.READ, description="Frames recorded without their tracklets")
    missed_poses:       Field[int]              = Field(0, access=Field.READ, description="Frames recorded without the poses asked for")
    dropped:            Field[int]              = Field(0, access=Field.READ, description="Frames dropped because the writing fell behind")


# The letter in a clip's file name: `0_C_001.mp4` is camera 0's video, chunk 1.
STOP_WAIT: float = 1.0          # s a stopping recording waits for the parts its last captures wait for
SPLIT_WAIT: float = 1.0         # s a split waits for a camera's first capture after its time, before it cuts that camera's chunk anyway
HELD: int = 1 << 30             # bytes waiting for the encoders, queued here and in the writers: 1 GB, some 6 s of four cameras with depth


def _latency(arrival: float | None, timestamp: float) -> float | None:
    return arrival - timestamp if arrival is not None else None


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
        self.frames: dict[int, Queue[RecordedCapture]] = {}
        self._held_lock: Lock = Lock()
        self._queued_bytes: int = 0                     # what the queues here hold
        self._dropped: int = 0                          # captures dropped whole for the bound, this recording
        self._unmerged: Thread | None = None            # what a crash left, merged beside the recording
        self.sync: RecordingSync = RecordingSync(self.with_depth)
        self.sync.add_capture_callback(self._on_capture)
        self.metadata: dict[int, TextIO] = {}
        self.metadata_frames: dict[int, int] = {}       # the frames of each camera's chunk so far
        self.chunks: dict[int, int] = {}                # each camera's chunk
        self.folder_path: Path | None = None

        for c in range(settings.num_cams):
            self.recorders[c] = {}
            self.frames[c] = Queue()
            for t in self.settings.video_frame_types:
                vcodec: str = DEPTH_ENCODER if t == FrameType.DEPTH else encoder_name(settings.video_format, settings.video_encoder)
                self.recorders[c][t] = StreamWriter(vcodec, settings.keyframe_interval, settings.video_quality)

        if FrameType.DEPTH in settings.video_frame_types and settings.keyframe_interval <= 0:
            logger.warning("keyframe_interval 0 with depth: each encoder picks its own keyframes, so seeking will show the wrong depth")

        self.suffix: str = settings.video_format.value
        self.split_requested = Event()
        self._split_at: float | None = None             # the capture time the chunks are cut after, while a split goes on
        self._split_request_time: float = float('-inf')
        self._newest: dict[int, float] = {}             # each camera's newest color frame handed in
        self._split_deadline: float = 0.0
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
        self.settings.unbind(RecorderSettings.start, self._on_start)
        self.settings.unbind(RecorderSettings.stop, self._on_stop)
        self.settings.unbind(RecorderSettings.split, self._on_split)
        self.settings.unbind(RecorderSettings.enabled, self._on_enabled)

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
                    self._begin()
            except Exception:
                logger.exception("Recorder error")

            time.sleep(0.01)

        if self._get_state() in (RecState.REC, RecState.STOP):
            self._set_state(RecState.STOP)
            self._stop_recording()
        self._set_state(RecState.IDLE)

    def _begin(self) -> None:
        """Start a recording; one that fails to start is given up once, and says why."""
        try:
            self._start_recording()
        except Exception:
            logger.exception("Recording could not start")
            for writers in self.recorders.values():
                for writer in writers.values():
                    writer.stop()
            self._close_metadata()
            self._set_state(RecState.IDLE)
            self.settings.recording = False
            return
        self._set_state(RecState.REC)

    def _stream_path(self, cam_id: int, frame_type: FrameType, chunk_index: int) -> Path:
        assert self.folder_path is not None
        suffix: str = DEPTH_SUFFIX if frame_type == FrameType.DEPTH else self.suffix
        return self.folder_path / make_file_name(cam_id, frame_type, chunk_index, suffix)

    def _metadata_path(self, cam_id: int, chunk_index: int) -> Path:
        assert self.folder_path is not None
        return self.folder_path / f"{cam_id}_{METADATA_LABEL}_{chunk_index:03d}{METADATA_SUFFIX}"

    def _open_metadata(self, cam_id: int) -> None:
        self.metadata[cam_id] = self._metadata_path(cam_id, self.chunks[cam_id]).open('w', encoding='utf-8')
        self.metadata_frames[cam_id] = 0

    def _close_metadata(self, cam_id: int | None = None) -> None:
        for c in [cam_id] if cam_id is not None else list(self.metadata):
            file: TextIO | None = self.metadata.pop(c, None)
            if file is not None:
                file.close()

    def _start_recording(self) -> None:
        output_path: Path = Path(self.data_path) / self.settings.output_path
        if self._unmerged is None or not self._unmerged.is_alive():        # what a crash left, beside the recording
            self._unmerged = Thread(target=finish_unmerged, args=(self.temp_path, output_path), daemon=True, name='RecorderUnmerged')
            self._unmerged.start()

        self.folder_name = make_folder_name(self.settings.name)
        self.folder_path = self.temp_path / self.folder_name
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.output_folder_path = output_path / self.folder_name
        self.output_folder_path.mkdir(parents=True, exist_ok=True)

        self.sync.reset()
        for queue in self.frames.values():          # a capture that came after the last recording's last write
            while True:
                try:
                    queue.get_nowait()
                except Empty:
                    break
        with self._held_lock:
            self._queued_bytes = 0
            self._dropped = 0
            self._newest.clear()
        self._split_at = None
        self.split_requested.clear()
        for c in range(self.settings.num_cams):
            self.chunks[c] = 0
            for t in self.settings.video_frame_types:
                self.recorders[c][t].start(str(self._stream_path(c, t, 0)), self.settings.fps)
            self._open_metadata(c)

        logger.info("VideoRecorder started to %s", self.output_folder_path)

    def _stop_recording(self) -> None:
        """No frame is taken any more; the captures waiting are settled by the tracklets and poses that still come,
        for at most `STOP_WAIT`, which only a stream that stopped with the recording runs out."""
        deadline: float = time.perf_counter() + STOP_WAIT
        while self.sync.waiting and time.perf_counter() < deadline:
            self._write_queued()
            time.sleep(0.01)
        self.sync.flush()
        self._write_queued()
        for c in range(self.settings.num_cams):
            for t in self.settings.video_frame_types:
                self.recorders[c][t].stop()

        self._close_metadata()
        for c in range(self.settings.num_cams):
            self._finish_chunk(c, self.chunks.get(c, 0))
        for merge in self.merges:
            merge.join()
        self.merges.clear()
        if self._unmerged is not None:
            self._unmerged.join()
        self._remove_temp_folder()
        self._update_missed()
        self.sync.missed.warn()
        logger.info("VideoRecorder stopped")

    def _split_camera(self, cam_id: int) -> None:
        """Cut a camera's chunk, on the recorder's thread, between its captures: its streams and its metadata go on
        in the next chunk together."""
        old_chunk: int = self.chunks[cam_id]
        self.chunks[cam_id] = old_chunk + 1
        for t in self.settings.video_frame_types:
            self.recorders[cam_id][t].split(str(self._stream_path(cam_id, t, old_chunk + 1)), self.settings.fps)
        self._close_metadata(cam_id)
        self._open_metadata(cam_id)
        self._finish_chunk(cam_id, old_chunk)

    def _finish_chunk(self, cam_id: int, chunk_index: int) -> None:
        if self.folder_path is None:
            return
        merge = Thread(target=self._merge_chunk, args=(cam_id, chunk_index))     # the next chunk must not wait for it
        merge.start()
        self.merges.append(merge)

    def _merge_chunk(self, cam_id: int, chunk_index: int) -> None:
        video: Path = self._stream_path(cam_id, FrameType.VIDEO, chunk_index)
        depth: Path = self._stream_path(cam_id, FrameType.DEPTH, chunk_index)
        metadata: Path = self._metadata_path(cam_id, chunk_index)
        if not (video.exists() and metadata.exists()):
            for part in (video, depth, metadata):
                part.unlink(missing_ok=True)    # a chunk that never got a frame
            return
        # A camera that sent no depth has none: its depth writer never got a frame, and opened no file
        merge_chunk(video, depth if depth.exists() else None, metadata, self.output_folder_path / make_chunk_name(cam_id, chunk_index))

    def _remove_temp_folder(self) -> None:
        if self.folder_path is not None and self.folder_path.exists():
            try:
                self.folder_path.rmdir()
            except OSError:
                logger.warning("Temp folder not empty, leaving: %s", self.folder_path)

    def _update_recording(self) -> None:
        if self.split_requested.is_set() and self._split_at is None:     # cut after the newest frame come when asked
            self.split_requested.clear()
            self._split_at = self._split_request_time
            self._split_deadline = time.perf_counter() + SPLIT_WAIT
        self._write_queued()
        if self._split_at is not None and time.perf_counter() > self._split_deadline:
            for c in range(self.settings.num_cams):         # a camera with no capture after the cut: cut anyway
                if self.chunks[c] == self._split_from:
                    self._split_camera(c)
            self._split_at = None
        self._update_missed()
        self.sync.missed.warn()
        failed: list[str] = [w.output_file for writers in self.recorders.values() for w in writers.values() if w.failed]
        if failed:
            logger.error("Recording ended: the encoder of %s died", ', '.join(failed))
            self._on_stop()

    @property
    def _split_from(self) -> int:
        """The chunk a split goes on from: the lowest of the cameras'."""
        return min(self.chunks.values(), default=0)

    def _update_missed(self) -> None:
        """The counts of what a recording went without, to the panel when they change."""
        depth, color, tracklets, poses = (self.sync.missed.total(part) for part in ('depth', 'color', 'tracklets', 'poses'))
        if self.settings.missed_depth != depth:
            self.settings.missed_depth = depth
        if self.settings.missed_color != color:
            self.settings.missed_color = color
        if self.settings.missed_tracklets != tracklets:
            self.settings.missed_tracklets = tracklets
        if self.settings.missed_poses != poses:
            self.settings.missed_poses = poses
        with self._held_lock:
            dropped: int = self._dropped
        if self.settings.dropped != dropped:
            self.settings.dropped = dropped

    def _write_queued(self) -> None:
        """Everything queued, on every pass: with depth two streams arrive per camera at the full frame rate, more
        than one frame per pass can keep up with. A camera's first capture after a split's time cuts its chunk."""
        split_from: int = self._split_from
        for c in range(self.settings.num_cams):
            while True:
                try:
                    capture: RecordedCapture = self.frames[c].get_nowait()
                except Empty:
                    break
                with self._held_lock:
                    self._queued_bytes -= capture.nbytes
                if self._split_at is not None and capture.video.timestamp > self._split_at and self.chunks[c] == split_from:
                    self._split_camera(c)
                self.recorders[c][FrameType.VIDEO].submit_frame(capture.video.image, capture.video.nv12)
                if capture.depth is not None:
                    self.recorders[c][FrameType.DEPTH].submit_frame(capture.depth.image)
                self._write_metadata(c, capture)
        if self._split_at is not None and all(chunk > split_from for chunk in self.chunks.values()):
            self._split_at = None           # every camera goes on in the new chunk

    def _write_metadata(self, cam_id: int, capture: RecordedCapture) -> None:
        file: TextIO | None = self.metadata.get(cam_id)
        if file is None:
            return
        frame: CameraFrame = capture.video
        count: int = self.metadata_frames[cam_id]
        if count == 0:
            height: int = frame.image.shape[0] * 2 // 3 if frame.nv12 else frame.image.shape[0]
            alignment = capture.depth.alignment if capture.depth is not None else None
            file.write(info_line(ChunkInfo(frame.image.shape[1], height, frame.intrinsics, frame.flip_h, self.settings.fps,
                                           alignment, capture.depth is not None)))
        tracklets: CameraTracklets | None = capture.tracklets
        poses: FramePoses | None = capture.poses
        file.write(frame_line(count, FrameMetadata(
            frame.timestamp, frame.readings, _latency(frame.arrival, frame.timestamp),
            tracklets.tracklets if tracklets is not None else None,
            _latency(tracklets.arrival, tracklets.timestamp) if tracklets is not None else None,
            poses.poses if poses is not None else None,
            _latency(poses.arrival, frame.timestamp) if poses is not None else None)))
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
        """A camera's frame callback: while recording, a frame of a kind the recorder writes to its sync,
        which puts each capture on the queue once it is complete."""
        if frame.kind in self.settings.video_frame_types and self._get_state() == RecState.REC:
            if frame.kind == FrameType.VIDEO:
                with self._held_lock:
                    self._newest[frame.cam_id] = max(frame.timestamp, self._newest.get(frame.cam_id, float('-inf')))
            self.sync.submit_frame(frame)

    def submit_tracklets(self, cam_tracklets: CameraTracklets) -> None:
        """A camera's tracklet callback, the tracklets as the camera gave them: while recording, to the sync.
        Tracklets found on the straightened image (the host detector's in -sim) are not a camera's, and stay out."""
        if self._taking_parts() and not cam_tracklets.straightened:
            self.sync.submit_tracklets(cam_tracklets)

    def expect_poses(self, cam_id: int, timestamp: float, asked: bool) -> None:
        """Whether the pose model was asked poses for the camera's frame captured at `timestamp`: while
        recording, to the sync, so a frame nobody asked poses for does not wait for them."""
        if self._taking_parts():
            self.sync.expect_poses(cam_id, timestamp, asked)

    def submit_poses(self, frame_poses: FramePoses) -> None:
        """The host's poses of one camera's frame: while recording, to the sync."""
        if self._taking_parts():
            self.sync.submit_poses(frame_poses)

    def _taking_parts(self) -> bool:
        """The tracklets and poses are taken while recording and while it stops: the captures waiting for them
        when the recording stops are settled by those of the frames after it."""
        return self._get_state() in (RecState.REC, RecState.STOP)

    def _on_capture(self, capture: RecordedCapture) -> None:
        """A capture the sync sent out, to the queue for the recorder's thread; dropped whole, and counted, while
        what waits for the encoders, here and in the writers, holds more than `HELD`."""
        writers: int = sum(w.queued_bytes for ws in self.recorders.values() for w in ws.values())
        with self._held_lock:
            if self._queued_bytes + writers + capture.nbytes > HELD:
                self._dropped += 1
                return
            self._queued_bytes += capture.nbytes
        self.frames[capture.video.cam_id].put(capture)

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
        """A split cuts after the newest frame any camera handed in by now: what came before it ends its chunk."""
        if self._get_state() == RecState.REC:
            with self._held_lock:
                self._split_request_time = max(self._newest.values(), default=float('-inf'))
            self.split_requested.set()
