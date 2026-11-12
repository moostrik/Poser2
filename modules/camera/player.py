"""Plays a folder of camera recordings: every camera's chunks in step on one clock, the recorded time
(`PlayClock`), each frame handed over as late after its recorded capture as it came live, a frame's
streams together, between in and out markers on the timeline, and the panel's playback readouts. It
feeds a source (the oak `Simulator` or the replay backend), which stamps the frames at hand-over. A
chunk is one `.mkv` with color, depth when it was recorded, and metadata (`chunk`), or a color-only
clip of before; a folder plays with the streams it has, whatever `video_frame_types` asks for.

The markers belong to the recording: each folder keeps its own in a file, so they survive a restart
and a change of preset. Dragging one pauses playback and shows the frame under it, the newest one
asked for as fast as a frame can be fetched; letting go plays from the in marker. Pressing the bar
beside them moves only the playhead: playback goes on from there, and a place outside the markers
shows its frame and plays from the in marker."""

import json
from threading import Thread, Event, Lock
from pathlib import Path
from numpy import ndarray
from typing import Callable, TypeAlias, cast
from enum import Enum, auto
from queue import Queue
from time import perf_counter, sleep

from modules.settings import BaseSettings, Field, Widget
from modules.video import CoderType, CoderFormat, hwaccel, StreamFrameCallback, StreamReader

from .chunk import ChunkInfo, ChunkMetadata, FrameMetadata, read_metadata
from .frame import FrameType
from .recorder import make_chunk_name, make_file_name

import logging
logger = logging.getLogger(__name__)


class PlayerSettings(BaseSettings):

    # ── Initial settings ───────────────────────────────────────────────
    num_cams:        Field[int]              = Field(1, access=Field.INIT, description="Number of cameras")
    fps:                Field[float]            = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Simulation FPS")
    video_path:         Field[str]              = Field("recordings", access=Field.INIT, description="Video recordings directory")
    video_decoder:      Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video decoder type")
    video_format:       Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, description="Video format")
    video_frame_types:  Field[list[FrameType]]  = Field([FrameType.VIDEO], access=Field.INIT, description="Frame types to record", visible=False)
    apply_warp:         Field[bool]             = Field(False, access=Field.INIT,
                                                        description="Run recordings through the camera's warp; only for raw (unwarped, tilt 0) clips")
    # Only clips recorded before the reprojection existed are raw: anything recorded since is already warped
    # (the equirectangular reprojection plus the tilt) and would be warped twice.

    # ── Playback controls ─────────────────────────────────────────────
    start:              Field[bool]             = Field(False, widget=Widget.button, description="Start playback", newline=True)
    stop:               Field[bool]             = Field(False, widget=Widget.button, description="Stop playback")
    available_folders:  Field[list[str]]        = Field([""], access=Field.READ, visible=False, description="Available recording folders")
    folder:             Field[str]              = Field("", widget=Widget.text_select, options=available_folders, description="Recording folder")
    refresh:            Field[bool]             = Field(False, widget=Widget.button, description="Refresh recording folders")
    sim_passthrough:    Field[bool]             = Field(False, description="Passthrough", visible=False)
    replay:             Field[bool]             = Field(False, access=Field.INIT, description="Play on the host without an OAK device: detector and tracker in software")
    playhead:           Field[list[float]]      = Field([0.0, 0.0], access=Field.READ, visible=False, description="Playhead and length of the recording (s)")
    scrub:              Field[float]            = Field(-1.0, visible=False, saved=False, description="Where the dragged marker is, as a fraction of the length; -1 when none is")
    seek:               Field[float]            = Field(-1.0, visible=False, saved=False, description="Where the bar was pressed, as a fraction of the length; -1 otherwise")
    in_out:             Field[list[float]]      = Field([0.0, 1.0], widget=Widget.timeline, options=playhead, drag=scrub, seek=seek, saved=False, label="In / Out", newline=True,
                                                        description="Plays between the two handles; each recording keeps its own")
    current_chunk:      Field[int]              = Field(0, access=Field.READ, description="Current chunk", newline=True)
    max_chunks:         Field[int]              = Field(0, access=Field.READ, description="Total chunks")


# A clip frame as the player hands it over: (cam_id, frame_type, image, metadata, timestamp), the
# metadata the chunk keeps of the frame and its capture on the play's clock (`PlayClock.capture`).
# Both are None for a recording without metadata: the source it feeds stamps the frame at its
# hand-over then.
ClipFrameCallback: TypeAlias = Callable[[int, FrameType, ndarray, FrameMetadata | None, float | None], None]
# Playback (re)starting at the in marker: the first play of a folder and every loop, not a scrub.
LoopCallback: TypeAlias = Callable[[], None]

IN_OUT_FILE: str = 'in_out.json'        # in a recording's folder: its markers


def read_in_out(folder: Path) -> list[float]:
    """A recording's markers; all of it when it has none, or they cannot be read."""
    path: Path = folder / IN_OUT_FILE
    if not path.is_file():
        return [0.0, 1.0]
    try:
        markers: dict = json.loads(path.read_text(encoding='utf-8'))
        return [float(markers['in']), float(markers['out'])]
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("Could not read the markers in %s", path)
        return [0.0, 1.0]


def write_in_out(folder: Path, in_out: list[float]) -> None:
    try:
        (folder / IN_OUT_FILE).write_text(json.dumps({'in': in_out[0], 'out': in_out[1]}), encoding='utf-8')
    except OSError:
        logger.exception("Could not save the markers in %s", folder)


class State(Enum):
    IDLE = auto()
    LOAD = auto()
    LOADING = auto()
    PLAYING = auto()
    STOP = auto()
    STOPPING = auto()
    NEXT = auto()
    SCRUB = auto()
    SEEKING = auto()

class MessageType(Enum):
    START = auto()
    STOP = auto()
    NEXT = auto()
    SCRUB = auto()
    RESUME = auto()
    SEEK = auto()

class Message():
    def __init__(self, state: MessageType, value: str = '', prepare: bool = False) -> None:
        self.state: MessageType = state
        self.value: str = value
        self.prepare: bool = prepare

class Folder():
    def __init__(self, name: str, path: Path, chunks: int) -> None:
        self.name: str = name
        self.path: Path = path
        self.chunks: int = chunks

FolderDict = dict[str, Folder]


class PlayClock:
    """The recorded time of one chunk's play, on `perf_counter`: the `FrameSchedule` every reader of the chunk
    shares. From `start` frame `n` of a camera is captured at the origin plus its recorded capture time minus the
    first frame's (`capture`), and due at that plus its recorded latency (`moment`), so a frame comes as late after
    its capture as it came live, and a frame lost live leaves its gap. A reader that falls behind moves the origin
    for every reader (`delay`). Frames the chunk has no time for are not on it: their readers pace at `fps`."""

    def __init__(self, frames: dict[int, list[FrameMetadata]], first: int) -> None:
        """`frames`: each camera's metadata by frame number; `first`: the frame number the play starts at."""
        self._frames: dict[int, list[FrameMetadata]] = frames
        starts: list[float] = [cam_frames[first].timestamp for cam_frames in frames.values() if first < len(cam_frames)]
        self._base: float | None = min(starts) if starts else None
        self._origin: float | None = None
        self._lock: Lock = Lock()

    def start(self, origin: float) -> None:
        with self._lock:
            self._origin = origin

    def delay(self, seconds: float) -> None:
        with self._lock:
            if self._origin is not None:
                self._origin += seconds

    def capture(self, cam_id: int, frame_id: int) -> float | None:
        """When the frame was captured, on the play's clock."""
        frame: FrameMetadata | None = self._frame(cam_id, frame_id)
        with self._lock:
            if frame is None or self._origin is None or self._base is None:
                return None
            return self._origin + frame.timestamp - self._base

    def moment(self, cam_id: int, frame_id: int) -> float | None:
        """When the frame is handed over: its capture plus the latency it came with live."""
        captured: float | None = self.capture(cam_id, frame_id)
        if captured is None:
            return None
        frame: FrameMetadata | None = self._frame(cam_id, frame_id)
        latency: float | None = frame.video_latency if frame is not None else None
        return captured + (latency or 0.0)

    def _frame(self, cam_id: int, frame_id: int) -> FrameMetadata | None:
        cam_frames: list[FrameMetadata] | None = self._frames.get(cam_id)
        if cam_frames is None or not 0 <= frame_id < len(cam_frames):
            return None
        return cam_frames[frame_id]

class Player(Thread):
    def __init__(self, settings: PlayerSettings, data_path: str = "", gray: bool = False) -> None:
        """``gray``: the consuming pipeline is mono, so video decodes straight to gray."""
        super().__init__()
        self.settings: PlayerSettings = settings
        self.data_path: str = data_path
        self.gray: bool = gray
        self.num_cams: int = settings.num_cams
        self.types: list[FrameType] = settings.video_frame_types
        self.fps: float = settings.fps

        self.running: bool = False
        self.state_messages: Queue[Message] = Queue()

        self.stop_event = Event()

        self.play_chunk: int = -1
        self.load_chunk: int = -1
        self.scrub_target: float = -1.0     # where the dragged marker is,
        self.scrub_shown: float = -1.0      # and the last place a frame was fetched for
        self.seek_target: float = 0.0       # where the bar was pressed,
        self.seek_outside: bool = False     # and whether that is outside the markers
        self.num_chunks: int = 0
        self.chunk_frames: list[int] = []
        self.start_chunk: int = 0
        self.start_offset: int = 0
        self.end_chunk: int = 0
        self.end_offset: int = 0
        self.load_folder: str = ''
        self.active_folder: str = ''        # folder we currently intend to play ('' = stopped)
        self._loading_markers: bool = False   # a load writes `in_out`: not a move to save
        self.suffix: str = settings.video_format.value

        self.folders: FolderDict = self._get_video_folders(settings, data_path)
        self.settings.available_folders = list(self.folders.keys())

        self.hwaccel, self.hwaccel_device = hwaccel(settings.video_decoder)

        self.players: list[StreamReader] = []
        self.loaders: list[StreamReader] = []
        self.closers: list[StreamReader] = []

        self.sync_lock: Lock = Lock()
        self.frame_sync_dict: dict[int, dict[int, dict[FrameType, ndarray]]] = {}
        for i in range(self.num_cams):
            self.frame_sync_dict[i] = {}
        self.load_types: set[FrameType] = set()     # the streams of the chunk being loaded,
        self.play_types: set[FrameType] = set()     # and of the one playing: what a frame waits for
        self.chunk_metadata: dict[tuple[int, int], ChunkMetadata] = {}      # by (cam_id, chunk)
        self.load_clock: PlayClock | None = None    # the recorded time of the chunk being loaded,
        self.play_clock: PlayClock | None = None    # and of the one playing

        self.drift_lock: Lock = Lock()
        self.drift_ids: dict[int, int] = {}
        self.drift:int = 0

        self.playback_lock: Lock = Lock()
        self.frame_callbacks: list[ClipFrameCallback] = []
        self.loop_callbacks: list[LoopCallback] = []
        self._gui_update_time: float = 0.0

        # Bind player settings callbacks
        self.settings.bind(PlayerSettings.start, self._on_start)
        self.settings.bind(PlayerSettings.stop, self._on_stop)
        self.settings.bind(PlayerSettings.folder, self._on_folder_changed)
        self.settings.bind(PlayerSettings.in_out, self._on_in_out)
        self.settings.bind(PlayerSettings.scrub, self._on_scrub)
        self.settings.bind(PlayerSettings.seek, self._on_seek)
        self.settings.bind(PlayerSettings.refresh, self._on_refresh_path)
        self.settings.bind(PlayerSettings.video_path, self._on_refresh_path)

    def stop(self) -> None:
        self._set_play_chunk(-1)
        message: Message = Message(MessageType.STOP)
        self.state_messages.put(message)
        self.stop_event.set()
        self.join()

    def run(self) -> None:
        state: State = State.IDLE
        self.running = True

        # Auto-start if a folder was loaded from preset
        folder = self.settings.folder
        if folder and folder in self.folders:
            self.play(True, folder, prepare=True)

        while self.running:
            try:
                message: Message | None = None
                try:
                    message = self.state_messages.get(block=False)
                except Exception as e:
                    message = None

                if message and message.state == MessageType.START:
                    if message.prepare:
                        self._prepare_folder(message.value)
                    self._set_load_folder(message.value)
                    self._set_load_chunk(-1)
                    self._set_play_chunk(-1)
                    state = State.LOAD
                if message and message.state == MessageType.STOP:
                    state = State.STOP
                if message and message.state == MessageType.SCRUB and state in (State.LOAD, State.LOADING, State.PLAYING):
                    self._stop()
                    self.scrub_shown = -1.0
                    state = State.SCRUB
                if message and message.state == MessageType.RESUME and state == State.SCRUB:
                    self._set_load_chunk(-1)
                    self._set_play_chunk(-1)
                    state = State.LOAD
                if message and message.state == MessageType.SEEK and state in (State.LOAD, State.LOADING, State.PLAYING, State.SEEKING):
                    self._stop()
                    self._load_seek()
                    state = State.SEEKING
                if state == State.SEEKING and self._finished_loading():
                    self._start(loop=False)
                    sleep(0.1) # waiting a bit reduces drift
                    if self.seek_outside:
                        self._set_load_chunk(self.start_chunk - 1)      # so the next load is the in marker's
                    self._load()
                    state = State.PLAYING
                if state == State.SCRUB:
                    self._update_scrub()
                if state == State.LOAD:
                    self._stop()
                    self._load()
                    state = State.LOADING
                if state == State.LOADING and self._finished_loading():
                    self._start()
                    sleep(0.1) # waiting a bit reduces drift
                    self._load()
                    state = State.PLAYING
                if state == State.PLAYING and self._finished_playing() and self._finished_loading():
                    self._start()
                    self._load()
                if state == State.STOP:
                    self._stop()
                    state = State.STOPPING
                if state == State.STOPPING and self._finished_stopping():
                    state = State.IDLE
                self._clean()

                if self.stop_event.is_set() and state == State.IDLE:
                    self.running = False
            except Exception:
                logger.exception("Player error")

            sleep(0.01)

    def _load(self) -> None:
        with self.playback_lock:
            start_chunk: int = self.start_chunk
            start_offset: int = self.start_offset
            end_chunk: int = self.end_chunk
            end_offset: int = self.end_offset

        # Advance to the next chunk in the [start_chunk, end_chunk] range, wrapping.
        LC: int = self._get_load_chunk() + 1
        if LC > end_chunk or LC < start_chunk:
            LC = start_chunk
        self._set_load_chunk(LC)
        self._load_chunk(start_offset if LC == start_chunk else 0, end_offset if LC == end_chunk else None)

    def _load_chunk(self, lo: int, hi: int | None) -> None:
        """The readers of the load chunk's frames `lo` up to `hi`."""
        folder: Folder = self.folders[self.load_folder]
        self.load_types = set()
        frames: dict[int, list[FrameMetadata]] = {}
        for c in range(self.num_cams):
            chunk_path: Path = folder.path / make_chunk_name(c, self.load_chunk)
            if chunk_path.is_file():
                metadata: ChunkMetadata = read_metadata(chunk_path, c)
                self.chunk_metadata[(c, self.load_chunk)] = metadata
                frames[c] = metadata.frames
        self.load_clock = PlayClock(frames, lo)
        for c in range(self.num_cams):
            chunk_path = folder.path / make_chunk_name(c, self.load_chunk)
            if chunk_path.is_file():
                info: ChunkInfo | None = self.chunk_metadata[(c, self.load_chunk)].info
                with_depth: bool = info is None or info.depth
                for stream, t in enumerate((FrameType.VIDEO, FrameType.DEPTH)):
                    if t in self.types and (t == FrameType.VIDEO or with_depth):
                        self._load_stream(c, t, chunk_path, stream, lo, hi, self.load_clock)
                continue
            for t in self.types:
                path: Path = folder.path / make_file_name(c, t, self.load_chunk, self.suffix)
                if path.is_file():
                    self._load_stream(c, t, path, 0, lo, hi, None)
                elif t == FrameType.VIDEO:
                    logger.warning(f"File {path} not found")

    def _load_stream(self, cam_id: int, frame_type: FrameType, path: Path, stream: int, lo: int, hi: int | None,
                     clock: PlayClock | None) -> None:
        color: bool = frame_type == FrameType.VIDEO and not self.gray
        player: StreamReader = StreamReader(cam_id, frame_type, cast(StreamFrameCallback, self._frame_sync_callback), color,
                                            self.hwaccel, self.hwaccel_device, self.fps, stream, frame_type == FrameType.DEPTH, clock)
        player.load(str(path), self.load_chunk, lo, hi)
        self.loaders.append(player)
        self.load_types.add(frame_type)

    def _start(self, loop: bool = True) -> None:
        """Play what is loaded. ``loop``: the loop callbacks may hear of it, at the in marker."""
        for p in self.players:
            p.stop()
            self.closers.append(p)
        self.players.clear()

        self._set_play_chunk(self._get_load_chunk())
        self._clear_frame_sync()
        self._clear_drift()
        with self.sync_lock:
            self.play_types = self.load_types
        clock: PlayClock | None = self.load_clock
        if clock is not None:
            clock.start(perf_counter())
        with self.playback_lock:
            self.play_clock = clock

        for p in self.loaders:
            p.play()
            self.players.append(p)
        self.loaders.clear()

        with self.playback_lock:
            at_in_marker: bool = self.load_chunk == self.start_chunk
        if at_in_marker and loop:
            for callback in self.loop_callbacks:
                try:
                    callback()
                except Exception:
                    logger.exception("Player loop callback failed")

    def _update_scrub(self) -> None:
        """Show one frame of the dragged position through the normal readers, then the newest
        position by then: a drag outruns the fetching, and the frames between do not matter."""
        if self.loaders:
            if self._finished_loading():
                self._start(loop=False)
            return
        with self.playback_lock:
            target: float = self.scrub_target
            chunk_frames: list[int] = self.chunk_frames
        total: int = sum(chunk_frames)
        if target == self.scrub_shown or total <= 0 or not self._finished_playing():
            return
        self.scrub_shown = target
        chunk, frame = self._locate(chunk_frames, min(round(target * total), total - 1))
        self._set_load_chunk(chunk)
        self._load_chunk(frame, frame + 1)

    def _load_seek(self) -> None:
        """The readers from the pressed place on, up to the out marker. Outside the markers there is
        only that frame to play."""
        with self.playback_lock:
            target: float = self.seek_target
            chunk_frames: list[int] = self.chunk_frames
            start_chunk, start_offset = self.start_chunk, self.start_offset
            end_chunk, end_offset = self.end_chunk, self.end_offset
        total: int = sum(chunk_frames)
        if total <= 0:
            return
        place: int = min(round(target * total), total - 1)
        chunk, frame = self._locate(chunk_frames, place)
        self.seek_outside = not sum(chunk_frames[:start_chunk]) + start_offset <= place < sum(chunk_frames[:end_chunk]) + end_offset
        self._set_load_chunk(chunk)
        if self.seek_outside:
            self._load_chunk(frame, frame + 1)
        else:
            self._load_chunk(frame, end_offset if chunk == end_chunk else None)

    def _stop(self) -> None:
        self._set_play_chunk(-1)
        for p in self.loaders:
            p.stop()
            self.closers.append(p)
        self.loaders.clear()
        for p in self.players:
            p.stop()
            self.closers.append(p)
        self.players.clear()

    def _clean(self) -> None:
        for p in self.closers:
            if p.is_stopped():
                p.join()
                self.closers.remove(p)

    def _finished_loading(self) -> bool:
        for p in self.loaders:
            if not p.is_loaded():
                return False
        return True

    def _finished_playing(self) -> bool:
        for p in self.players:
            if p.is_playing():
                return False
        return True

    def _finished_stopping(self) -> bool:
        for p in self.closers:
            if not p.is_stopped():
                return False
        return True

    # FRAME CALLBACK
    def _frame_sync_callback(self, cam_id: int, frame_type: FrameType, frame: ndarray, chunk_id: int, frame_id: int) -> None:
        if chunk_id != self._get_play_chunk():
            # print(f"Chunk {chunk_id} does not match load chunk {self.get_current_chunk()}, ignoring frame")
            return

        if cam_id == 0 and frame_type == FrameType.VIDEO:
            self.update_gui(frame_id)

        sync_frames:dict[FrameType, ndarray] = {}
        with self.sync_lock:
            cam_frames: dict[int, dict[FrameType, ndarray]] = self.frame_sync_dict[cam_id]
            if frame_id not in cam_frames:
                cam_frames[frame_id] = {}
            cam_frames[frame_id][frame_type] = frame
            for t in self.play_types:
                if t not in cam_frames[frame_id]:
                    return

            keys_to_delete: list[int] = []

            for key in cam_frames.keys():
                if key < frame_id - 3:
                    logger.info(f"Frame {key} is older than {frame_id}, marking for removal")
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del cam_frames[key]

            sync_frames = cam_frames[frame_id]
            del cam_frames[frame_id]

        # make thread safe and reset on start of new chunk
        self._calcuate_drift(cam_id, frame_id)

        metadata: ChunkMetadata | None = self.chunk_metadata.get((cam_id, chunk_id))
        frame_metadata: FrameMetadata | None = None
        if metadata is not None and frame_id < len(metadata.frames):
            frame_metadata = metadata.frames[frame_id]
        with self.playback_lock:
            clock: PlayClock | None = self.play_clock
        timestamp: float | None = clock.capture(cam_id, frame_id) if clock is not None else None
        for ft in self.types:       # VIDEO first: its readings set up what the depth of the frame uses
            if ft in sync_frames:
                for callback in self.frame_callbacks:
                    callback(cam_id, ft, sync_frames[ft], frame_metadata, timestamp)

    def _clear_frame_sync(self) -> None:
        with self.sync_lock:
            for i in range(self.num_cams):
                self.frame_sync_dict[i].clear()

    # DRIFT CALCULATION
    def _calcuate_drift(self, cam_id, frame_id: int) -> None:
        drift_ids: dict[int, int] = {}
        with self.drift_lock:
            self.drift_ids[cam_id] = frame_id
            drift_ids = self.drift_ids.copy()

        max_id: int = 1000000
        low_id: int = max_id
        high_id: int = 0

        for key in drift_ids.keys():
            if drift_ids[key] < low_id:
                low_id = drift_ids[key]
            if drift_ids[key] > high_id:
                high_id = drift_ids[key]

        drift:int = 0
        if low_id == max_id:
            drift = 0
        else:
            drift = high_id - low_id

        with self.drift_lock:
            self.drift = drift

    def _clear_drift(self) -> None:
        with self.drift_lock:
            self.drift_ids.clear()
            self.drift = 0

    # GETTERS AND SETTERS
    def _set_load_chunk(self, value: int) -> None:
        with self.playback_lock:
            self.load_chunk = value

    def _get_load_chunk(self) -> int:
        with self.playback_lock:
            return self.load_chunk

    def _set_play_chunk(self, value: int) -> None:
        with self.playback_lock:
            self.play_chunk = value

    def _get_play_chunk(self) -> int:
        with self.playback_lock:
            return self.play_chunk

    def _set_load_folder(self, value: str) -> None:
        with self.playback_lock:
            self.load_folder = value

    def _get_load_folder(self) -> str:
        with self.playback_lock:
            return self.load_folder

    # EXTERNAL METHODS
    def play(self, value: bool, name: str = '', prepare: bool = False) -> None:
        """``prepare``: probe the folder's chunk frame counts on the player thread before loading."""
        if value:
            if not name in self.folders:
                logger.warning(f"Folder {name} not found")
                return
            message: Message = Message(MessageType.START, name, prepare)
            self.active_folder = name
        else:
            message: Message = Message(MessageType.STOP)
            self.active_folder = ''

        self.num_chunks = self.get_num_folder_chunks(name)
        self.state_messages.put(message)

    def get_folder_names(self) -> list[str]:
        return list(self.folders.keys())

    def get_num_folder_chunks(self, folder: str) -> int:
        return self.folders[folder].chunks if folder in self.folders else 0

    def get_current_chunk(self) -> int:
        return self._get_play_chunk()

    def get_current_folder(self) -> str:
        return self._get_load_folder()

    def get_num_chunks(self) -> int:
        return self.num_chunks

    def get_drift(self) -> int:
        with self.drift_lock:
            return self.drift

    def get_chunk_info(self, cam_id: int) -> ChunkInfo | None:
        """The fixed info of the chunk playing for a camera; None for a recording without it."""
        metadata: ChunkMetadata | None = self.chunk_metadata.get((cam_id, self._get_play_chunk()))
        return metadata.info if metadata is not None else None

    # CALLBACKS
    def add_frame_callback(self, callback: ClipFrameCallback) -> None:
        if self.running:
            logger.warning('Player: cannot add callback while it is running')
            return
        if callback not in self.frame_callbacks:
            self.frame_callbacks.append(callback)

    def add_loop_callback(self, callback: LoopCallback) -> None:
        """Called on the player thread when playback starts at the in marker: keep it short."""
        if self.running:
            logger.warning('Player: cannot add callback while it is running')
            return
        if callback not in self.loop_callbacks:
            self.loop_callbacks.append(callback)

    # STATIC METHODS
    @staticmethod
    def _get_video_folders(settings: PlayerSettings, data_path: str = "") -> FolderDict:
        folders: FolderDict = {}
        video_path: Path = Path(data_path) / settings.video_path
        if not video_path.is_dir():
            return folders
        for folder in video_path.iterdir():
            if folder.is_dir():
                max_chunk: int = -1
                for file in folder.iterdir():
                    if file.is_file() and file.suffix in ('.mkv', CoderFormat.H264.value, CoderFormat.H265.value):
                        n: str = file.stem.split('_')[-1]       # `0_000.mkv`, or `0_C_000.mp4` of a colour-only clip
                        if n.isdigit():
                            max_chunk = max(max_chunk, int(n))
                if max_chunk >= 0:
                    folders[folder.name] = (Folder(folder.name, folder, max_chunk))
        return folders

    # SETTINGS CALLBACKS
    def _on_start(self, _=None) -> None:
        folder: str = self.settings.folder
        if folder:
            self.play(True, folder)

    def _on_stop(self, _=None) -> None:
        self.play(False, '')

    def _on_folder_changed(self, folder: str) -> None:
        if folder and folder in self.folders:
            # prepare on the player thread: the probe decodes whole clips and this callback
            # runs on the GUI event loop, which must not stall past the socket heartbeat
            self.play(True, folder, prepare=True)

    def _on_refresh_path(self, _=None) -> None:
        self.folders = self._get_video_folders(self.settings, self.data_path)
        self.settings.available_folders = list(self.folders.keys())

    def _on_in_out(self, _=None) -> None:
        self._recompute_range()
        if self._loading_markers:
            return
        folder: Folder | None = self.folders.get(self.settings.folder)
        if folder is not None:
            write_in_out(folder.path, self.settings.in_out)

    def _on_scrub(self, value: float) -> None:
        """A marker dragged, or let go at -1. Without a recording playing there is nothing to pause."""
        if not self.active_folder:
            return
        if value < 0.0:
            self.state_messages.put(Message(MessageType.RESUME))
            return
        with self.playback_lock:
            self.scrub_target = value
        self.state_messages.put(Message(MessageType.SCRUB))

    def _on_seek(self, value: float) -> None:
        """The bar pressed, or back at -1. Without a recording playing there is nothing to move."""
        if value < 0.0 or not self.active_folder:
            return
        with self.playback_lock:
            self.seek_target = value
        self.state_messages.put(Message(MessageType.SEEK))

    def update_gui(self, frame_id: int = 0) -> None:
        # Settings writes run share propagation and callbacks on this (reader) thread; a GUI
        # readout needs no more than 4 Hz
        now: float = perf_counter()
        if now - self._gui_update_time < 0.25:
            return
        self._gui_update_time = now
        chunk: int = self.get_current_chunk()
        self.settings.current_chunk = chunk
        with self.playback_lock:
            total: int = sum(self.chunk_frames)
        self.settings.playhead = [self._global_frame(chunk, frame_id) / self.fps, total / self.fps]

    def _global_frame(self, chunk: int, frame_id: int) -> int:
        """Absolute frame index across the whole timeline for (chunk, in-chunk frame)."""
        if chunk < 0:
            return 0
        with self.playback_lock:
            chunk_frames: list[int] = self.chunk_frames
        return sum(chunk_frames[:chunk]) + frame_id

    # RANGE / TIMELINE
    def _prepare_folder(self, folder: str) -> None:
        """Probe every chunk's frame count and take up the recording's markers; the caller starts playback."""
        chunk_frames: list[int] = self._probe_chunk_frames(folder)
        with self.playback_lock:
            self.chunk_frames = chunk_frames
        self.settings.max_chunks = len(chunk_frames)
        self.settings.playhead = [0.0, sum(chunk_frames) / self.fps]
        self._loading_markers = True
        try:
            self.settings.in_out = read_in_out(self.folders[folder].path)
        finally:
            self._loading_markers = False
        self._recompute_range()

    def _probe_chunk_frames(self, folder: str) -> list[int]:
        """Frame count of each chunk (cam 0 / first frame type), indexed 0..max_chunk."""
        if folder not in self.folders:
            return []
        folder_obj: Folder = self.folders[folder]
        frame_type: FrameType = self.types[0]
        frames: list[int] = []
        for chunk in range(folder_obj.chunks + 1):
            path: Path = folder_obj.path / make_chunk_name(0, chunk)
            if not path.is_file():
                path = folder_obj.path / make_file_name(0, frame_type, chunk, self.suffix)
            count: int = StreamReader.frame_count(str(path)) if path.is_file() else 0
            frames.append(count)
        return frames

    def _recompute_range(self) -> None:
        """Map the normalized markers onto global frame positions and chunk offsets."""
        with self.playback_lock:
            chunk_frames: list[int] = self.chunk_frames
            total: int = sum(chunk_frames)

        if total <= 0:
            with self.playback_lock:
                self.start_chunk = self.start_offset = 0
                self.end_chunk = self.end_offset = 0
            return

        a, b = (max(0.0, min(1.0, marker)) for marker in self.settings.in_out)
        start_norm, end_norm = min(a, b), max(a, b)

        start_global: int = min(max(round(start_norm * total), 0), total - 1)
        end_global: int = min(max(round(end_norm * total), start_global + 1), total)

        start_chunk, start_offset = self._locate(chunk_frames, start_global)
        end_chunk, end_offset = self._locate_end(chunk_frames, end_global)

        with self.playback_lock:
            self.start_chunk = start_chunk
            self.start_offset = start_offset
            self.end_chunk = end_chunk
            self.end_offset = end_offset

    @staticmethod
    def _locate(chunk_frames: list[int], global_frame: int) -> tuple[int, int]:
        """Map an (inclusive) global frame index to (chunk, offset within chunk)."""
        acc: int = 0
        for i, n in enumerate(chunk_frames):
            if global_frame < acc + n:
                return i, global_frame - acc
            acc += n
        last: int = len(chunk_frames) - 1
        return last, max(0, chunk_frames[last] - 1)

    @staticmethod
    def _locate_end(chunk_frames: list[int], end_global: int) -> tuple[int, int]:
        """Map an (exclusive) global end index to (chunk, exclusive offset in chunk)."""
        acc: int = 0
        for i, n in enumerate(chunk_frames):
            if end_global <= acc + n:
                return i, end_global - acc
            acc += n
        last: int = len(chunk_frames) - 1
        return last, chunk_frames[last]
