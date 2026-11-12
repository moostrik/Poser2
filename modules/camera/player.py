"""Plays a folder of camera recordings: every camera's chunks in step on one clock, the recorded time
(`PlayClock`), each frame handed over as late after its recorded capture as it came live, a frame's
streams together (`FramePairing`), between in and out markers on the timeline, and the panel's playback
readouts. It feeds a source (the oak `Simulator` or the replay backend). A chunk is one `.mkv` with color,
depth when it was recorded, and metadata (`chunk`), or a color-only clip of before; a folder plays with the
streams it has, whatever `video_frame_types` asks for. A chunk's readers are `ChunkReaders`; the folders,
their markers and the timeline are `RecordingFolder` and `Timeline`.

The markers belong to the recording: each folder keeps its own in a file, so they survive a restart
and a change of preset. Dragging one pauses playback and shows the frame under it, the newest one
asked for as fast as a frame can be fetched; letting go plays from the in marker. Pressing the bar
beside them moves only the playhead: playback goes on from there, and a place outside the markers
shows its frame and plays from the in marker."""

from threading import Thread, Event, Lock
from pathlib import Path
from typing import Callable, TypeAlias
from enum import Enum, auto
from queue import Queue, Empty
from time import perf_counter, sleep

from modules.settings import BaseSettings, Field, Widget
from modules.video import CoderType, CoderFormat, hwaccel

from .chunk import ChunkInfo
from .chunk_readers import ChunkReaders, Decoding
from .frame import FrameType
from .frame_pairing import ClipFrameCallback, FramePairing
from .recording_folder import RecordingFolder, Timeline, locate

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


# Playback (re)starting at the in marker: the first play of a folder and every loop, not a scrub.
LoopCallback: TypeAlias = Callable[[], None]


class State(Enum):
    IDLE = auto()
    LOAD = auto()
    LOADING = auto()
    PLAYING = auto()
    STOP = auto()
    STOPPING = auto()
    SCRUB = auto()
    SEEKING = auto()

class MessageType(Enum):
    START = auto()
    STOP = auto()
    SCRUB = auto()
    RESUME = auto()
    SEEK = auto()

class Message():
    def __init__(self, state: MessageType, value: str = '', prepare: bool = False) -> None:
        self.state: MessageType = state
        self.value: str = value
        self.prepare: bool = prepare


class Player(Thread):
    def __init__(self, settings: PlayerSettings, data_path: str = "", gray: bool = False, nv12: bool = False) -> None:
        """``gray``: the consuming pipeline is mono, so video decodes straight to gray. ``nv12``: color decodes to
        packed NV12, as a camera delivers it to an uploader that converts on the GPU (and half the bytes of BGR)."""
        super().__init__()
        self.settings: PlayerSettings = settings
        self.data_path: str = data_path
        self.gray: bool = gray
        self.nv12: bool = nv12 and not gray
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
        self.timeline: Timeline = Timeline()
        self.load_folder: str = ''
        self.active_folder: str = ''        # folder we currently intend to play ('' = stopped)
        self._loading_markers: bool = False   # a load writes `in_out`: not a move to save
        self.suffix: str = settings.video_format.value

        self.folders: dict[str, RecordingFolder] = RecordingFolder.find(Path(data_path) / settings.video_path)
        self.settings.available_folders = list(self.folders.keys())

        self.hwaccel, self.hwaccel_device = hwaccel(settings.video_decoder)

        self.loading: ChunkReaders | None = None        # the readers of the chunk being loaded,
        self.playing: ChunkReaders | None = None        # of the one playing,
        self.ending: ChunkReaders | None = None         # of the one before it, its last frames while the next plays,
        self.closing: list[ChunkReaders] = []           # and of those stopped, until their threads end
        self.known: ChunkReaders | None = None          # the last chunk played: its metadata for a seek after a stop
        self.pairing: FramePairing = FramePairing(self.num_cams, self.types)

        self.playback_lock: Lock = Lock()
        self.loop_callbacks: list[LoopCallback] = []
        self.jump_callbacks: list[LoopCallback] = []
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

    @property
    def frame_callbacks(self) -> list[ClipFrameCallback]:
        return self.pairing.frame_callbacks

    def stop(self) -> None:
        self._set_play_chunk(-1)
        self.state_messages.put(Message(MessageType.STOP))
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
                except Empty:
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
                    if self.seek_outside:
                        self._set_load_chunk(self._timeline().start_chunk - 1)      # so the next load is the in marker's
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
                    self._load()
                    state = State.PLAYING
                if state == State.PLAYING and self._follows() and self._finished_loading():
                    self._follow()
                    self._load()
                if state == State.PLAYING and self._finished_playing() and self._finished_loading():
                    self._start(follows=True)
                    self._load()
                self._retire()
                if state == State.STOP:
                    self._stop()
                    state = State.STOPPING
                if state == State.STOPPING and self._finished_stopping():
                    state = State.IDLE
                self._clean()
                self._update_readouts()

                if self.stop_event.is_set() and state == State.IDLE:
                    self.running = False
            except Exception:
                logger.exception("Player error")

            sleep(0.01)

    # ── The chunks' readers ───────────────────────────────────────────

    def _load(self) -> None:
        """The readers of the next chunk between the markers, wrapping to the in marker's."""
        timeline: Timeline = self._timeline()
        chunk: int = self._get_load_chunk() + 1
        if chunk > timeline.end_chunk or chunk < timeline.start_chunk:
            chunk = timeline.start_chunk
        self._set_load_chunk(chunk)
        self._load_chunk(timeline.start_offset if chunk == timeline.start_chunk else 0,
                         timeline.end_offset if chunk == timeline.end_chunk else None)

    def _load_chunk(self, lo: int, hi: int | None) -> None:
        """The readers of the load chunk's frames `lo` up to `hi`."""
        known: ChunkReaders | None = self.playing or self.loading or self.known
        decoding = Decoding(self.gray, self.nv12, self.hwaccel, self.hwaccel_device, self.fps, self.suffix)
        self.loading = ChunkReaders(self.folders[self.load_folder], self._get_load_chunk(), lo, hi, self.num_cams, self.types,
                                    self.pairing.submit_frame, decoding, known)

    def _start(self, loop: bool = True, follows: bool = False) -> None:
        """Play what is loaded. ``loop``: the loop callbacks may hear of it, at the in marker. ``follows``: it is
        the chunk after the one that played: its clock carries the playing one's on, and no jump is told."""
        before: ChunkReaders | None = self.playing
        if before is not None:
            before.stop()
            self.closing.append(before)
            self.known = before
        self._close_ending()
        readers: ChunkReaders | None = self.loading
        self.loading = None
        self.playing = readers
        runs_on: bool = follows and before is not None and readers is not None and readers.chunk == before.chunk + 1
        self._set_play_chunk(readers.chunk if readers is not None else -1)
        if readers is not None:
            if runs_on:
                assert before is not None
                readers.clock.carry_on(before.clock, perf_counter())
            else:
                readers.clock.start(perf_counter())
        self.pairing.start(readers)

        if not runs_on:
            for callback in self.jump_callbacks:
                try:
                    callback()
                except Exception:
                    logger.exception("Player jump callback failed")

        if readers is not None:
            readers.play()

        if readers is not None and readers.chunk == self._timeline().start_chunk and loop:
            for callback in self.loop_callbacks:
                try:
                    callback()
                except Exception:
                    logger.exception("Player loop callback failed")

    def _follows(self) -> bool:
        """The chunk playing has a reader ended and the one loading is the next in the recording: it starts beside
        the rest of the playing one, so no camera waits for the last to end."""
        playing, loading = self.playing, self.loading
        return (playing is not None and loading is not None and self.ending is None and playing.ending()
                and not playing.finished() and loading.folder is playing.folder and loading.chunk == playing.chunk + 1)

    def _follow(self) -> None:
        """Play the loaded next chunk on the playing one's clock, carried on, beside the playing one's last frames."""
        before: ChunkReaders | None = self.playing
        readers: ChunkReaders | None = self.loading
        assert before is not None and readers is not None
        self.loading = None
        self.ending = before
        self.playing = readers
        self._set_play_chunk(readers.chunk)
        readers.clock.carry_on(before.clock, perf_counter())
        self.pairing.follow(readers)
        readers.play()

    def _retire(self) -> None:
        """The chunk ending, once all its readers have ended: no more of its frames go out."""
        if self.ending is not None and self.ending.finished():
            self._close_ending()

    def _close_ending(self) -> None:
        ending: ChunkReaders | None = self.ending
        if ending is None:
            return
        self.ending = None
        self.pairing.retire(ending)
        ending.stop()
        self.closing.append(ending)

    def _update_scrub(self) -> None:
        """Show one frame of the dragged position through the normal readers, then the newest
        position by then: a drag outruns the fetching, and the frames between do not matter."""
        if self.loading is not None:
            if self._finished_loading():
                self._start(loop=False)
            return
        with self.playback_lock:
            target: float = self.scrub_target
        timeline: Timeline = self._timeline()
        if target == self.scrub_shown or timeline.total <= 0 or not self._finished_playing():
            return
        self.scrub_shown = target
        chunk, frame = locate(timeline.chunk_frames, timeline.place(target))
        self._set_load_chunk(chunk)
        self._load_chunk(frame, frame + 1)

    def _load_seek(self) -> None:
        """The readers from the pressed place on, up to the out marker. Outside the markers there is
        only that frame to play."""
        with self.playback_lock:
            target: float = self.seek_target
        timeline: Timeline = self._timeline()
        if timeline.total <= 0:
            return
        place: int = timeline.place(target)
        chunk, frame = locate(timeline.chunk_frames, place)
        self.seek_outside = not timeline.within_markers(place)
        self._set_load_chunk(chunk)
        if self.seek_outside:
            self._load_chunk(frame, frame + 1)
        else:
            self._load_chunk(frame, timeline.end_offset if chunk == timeline.end_chunk else None)

    def _stop(self) -> None:
        self._set_play_chunk(-1)
        self.pairing.start(None)
        self.known = self.playing or self.loading or self.known
        self._close_ending()
        for readers in (self.loading, self.playing):
            if readers is not None:
                readers.stop()
                self.closing.append(readers)
        self.loading = None
        self.playing = None

    def _clean(self) -> None:
        for readers in [r for r in self.closing if r.stopped()]:
            readers.join()
            self.closing.remove(readers)

    def _finished_loading(self) -> bool:
        return self.loading is None or self.loading.loaded()

    def _finished_playing(self) -> bool:
        return self.playing is None or self.playing.finished()

    def _finished_stopping(self) -> bool:
        return all(readers.stopped() for readers in self.closing)

    def _update_readouts(self) -> None:
        """On the player's thread: the panel's playback readouts, and the frames dropped unpaired since the last."""
        if self.pairing.shown_frame >= 0:
            self.update_gui(self.pairing.shown_frame)
        unpaired: int = self.pairing.take_unpaired()
        if unpaired:
            logger.info("Player: %d frames dropped without their other stream", unpaired)

    # ── Getters and setters ───────────────────────────────────────────

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

    def _timeline(self) -> Timeline:
        with self.playback_lock:
            return self.timeline

    # ── The app's side ────────────────────────────────────────────────

    def play(self, value: bool, name: str = '', prepare: bool = False) -> None:
        """``prepare``: probe the folder's chunk frame counts on the player thread before loading."""
        if value:
            if name not in self.folders:
                logger.warning(f"Folder {name} not found")
                return
            message: Message = Message(MessageType.START, name, prepare)
            self.active_folder = name
        else:
            message = Message(MessageType.STOP)
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

    def get_drift(self) -> float:
        """How far apart the cameras' last frames gone out were captured, in ms."""
        return self.pairing.drift

    def get_chunk_info(self, cam_id: int) -> ChunkInfo | None:
        """The fixed info of the chunk a camera's frames play from; None for a recording without it."""
        return self.pairing.info(cam_id)

    def add_frame_callback(self, callback: ClipFrameCallback) -> None:
        if self.running:
            logger.warning('Player: cannot add callback while it is running')
            return
        if callback not in self.pairing.frame_callbacks:
            self.pairing.frame_callbacks.append(callback)

    def add_loop_callback(self, callback: LoopCallback) -> None:
        """Called on the player thread when playback starts at the in marker: keep it short."""
        if self.running:
            logger.warning('Player: cannot add callback while it is running')
            return
        if callback not in self.loop_callbacks:
            self.loop_callbacks.append(callback)

    def add_jump_callback(self, callback: LoopCallback) -> None:
        """Called on the player thread when playback starts somewhere that does not follow what played: a loop, a
        seek, a scrub's frame, another folder. What waits for the frames before it is to be dropped: keep it short."""
        if self.running:
            logger.warning('Player: cannot add callback while it is running')
            return
        if callback not in self.jump_callbacks:
            self.jump_callbacks.append(callback)

    # ── Settings callbacks ────────────────────────────────────────────

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
        self.folders = RecordingFolder.find(Path(self.data_path) / self.settings.video_path)
        self.settings.available_folders = list(self.folders.keys())

    def _on_in_out(self, _=None) -> None:
        with self.playback_lock:
            self.timeline = Timeline.of(self.timeline.chunk_frames, self.settings.in_out)
        if self._loading_markers:
            return
        folder: RecordingFolder | None = self.folders.get(self.settings.folder)
        if folder is not None:
            folder.write_markers(self.settings.in_out)

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
        """The panel's playhead and chunk, on the player's thread, at most 4 times a second: a settings write runs
        its share propagation and callbacks on the writing thread, never a reader's."""
        now: float = perf_counter()
        if now - self._gui_update_time < 0.25:
            return
        self._gui_update_time = now
        chunk: int = self.get_current_chunk()
        self.settings.current_chunk = chunk
        timeline: Timeline = self._timeline()
        self.settings.playhead = [timeline.global_frame(chunk, frame_id) / self.fps, timeline.total / self.fps]

    def _prepare_folder(self, name: str) -> None:
        """Probe every chunk's frame count and take up the recording's markers; the caller starts playback."""
        folder: RecordingFolder | None = self.folders.get(name)
        chunk_frames: list[int] = folder.frame_counts(self.types[0], self.suffix) if folder is not None else []
        with self.playback_lock:
            self.timeline = Timeline.of(chunk_frames, self.settings.in_out)
        self.settings.max_chunks = len(chunk_frames)
        self.settings.playhead = [0.0, sum(chunk_frames) / self.fps]
        if folder is None:
            return
        self._loading_markers = True
        try:
            self.settings.in_out = folder.read_markers()
        finally:
            self._loading_markers = False
        with self.playback_lock:
            self.timeline = Timeline.of(chunk_frames, self.settings.in_out)
