from threading import Thread, Event, Lock
from pathlib import Path
from numpy import ndarray
from typing import Set, Dict
from enum import Enum, auto
from queue import Queue
from time import sleep

from ..camera.definitions import CoderType, CoderFormat, FrameType, FrameCallback
from .settings import SimulatorSettings
from .stream_reader import StreamReader
from ..recorder.recorder import make_file_name

import logging
logger = logging.getLogger(__name__)

class State(Enum):
    IDLE = auto()
    LOAD = auto()
    LOADING = auto()
    PLAYING = auto()
    STOP = auto()
    STOPPING = auto()
    NEXT = auto()

HwaccelString: dict[CoderType, str] = {
    CoderType.CPU:  '',
    CoderType.GPU:  'd3d12va',
    CoderType.iGPU: 'd3d12va'
}

HwaccelDeviceString: dict[CoderType, str] = {
    CoderType.CPU:  '',
    CoderType.GPU:  '0',
    CoderType.iGPU: '1'
}

class MessageType(Enum):
    START = auto()
    STOP = auto()
    NEXT = auto()

class Message():
    def __init__(self, state: MessageType, value: str = '') -> None:
        self.state: MessageType = state
        self.value: str = value

class Folder():
    def __init__(self, name: str, path: Path, chunks: int) -> None:
        self.name: str = name
        self.path: Path = path
        self.chunks: int = chunks

FolderDict = Dict[str, Folder]

class Player(Thread):
    def __init__(self, settings: SimulatorSettings, data_path: str = "") -> None:
        super().__init__()
        self.settings: SimulatorSettings = settings
        self.data_path: str = data_path
        self.num_cams: int = settings.num_cameras
        self.types: list[FrameType] = settings.video_frame_types
        self.fps: float = settings.fps

        self.running: bool = False
        self.state_messages: Queue[Message] = Queue()

        self.stop_event = Event()

        self.play_chunk: int = -1
        self.load_chunk: int = -1
        self.num_chunks: int = 0
        self.chunk_frames: list[int] = []
        self.start_chunk: int = 0
        self.start_offset: int = 0
        self.end_chunk: int = 0
        self.end_offset: int = 0
        self.load_folder: str = ''
        self.active_folder: str = ''        # folder we currently intend to play ('' = stopped)
        self._suppress_restart: bool = False  # guards programmatic norm resets
        self.suffix: str = settings.video_format.value

        self.folders: FolderDict = self._get_video_folders(settings, data_path)
        self.settings.available_folders = list(self.folders.keys())

        self.hwt: str = HwaccelString[settings.video_decoder]
        self.hwd: str = HwaccelDeviceString[settings.video_decoder]

        self.players: list[StreamReader] = []
        self.loaders: list[StreamReader] = []
        self.closers: list[StreamReader] = []

        self.sync_lock: Lock = Lock()
        self.frame_sync_dict: Dict[int, Dict[int, Dict[FrameType, ndarray]]] = {}
        for i in range(self.num_cams):
            self.frame_sync_dict[i] = {}

        self.drift_lock: Lock = Lock()
        self.drift_ids: Dict[int, int] = {}
        self.drift:int = 0

        self.playback_lock: Lock = Lock()
        self.frameCallbacks: Set[FrameCallback] = set()

        # Bind player settings callbacks
        self.settings.bind(SimulatorSettings.start, self._on_start)
        self.settings.bind(SimulatorSettings.stop, self._on_stop)
        self.settings.bind(SimulatorSettings.folder, self._on_folder_changed)
        self.settings.bind(SimulatorSettings.start_norm, self._on_range_changed)
        self.settings.bind(SimulatorSettings.end_norm, self._on_range_changed)
        self.settings.bind(SimulatorSettings.refresh, self._on_refresh_path)
        self.settings.bind(SimulatorSettings.video_path, self._on_refresh_path)

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
            self._prepare_folder(folder)
            self.play(True, folder)

        while self.running:
            try:
                message: Message | None = None
                try:
                    message = self.state_messages.get(block=False)
                except Exception as e:
                    message = None

                if message and message.state == MessageType.START:
                    self._set_load_folder(message.value)
                    self._set_load_chunk(-1)
                    self._set_play_chunk(-1)
                    state = State.LOAD
                if message and message.state == MessageType.STOP:
                    state = State.STOP
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
        folder: Folder = self.folders[self.load_folder]

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

        lo: int = start_offset if LC == start_chunk else 0
        hi: int | None = end_offset if LC == end_chunk else None

        for c in range(self.num_cams):
            for t in self.types:
                path: Path = folder.path / make_file_name(c, t, self.load_chunk, self.suffix)
                if path.is_file():

                    player: StreamReader = StreamReader(c, t, self._frame_sync_callback, self.hwt, self.hwd, self.fps)
                    player.load(str(path), self.load_chunk, lo, hi)
                    self.loaders.append(player)
                else:
                    logger.warning(f"File {path} not found")

    def _start(self) -> None:
        for p in self.players:
            p.stop()
            self.closers.append(p)
        self.players.clear()

        self._set_play_chunk(self._get_load_chunk())
        self._clear_frame_sync()
        self._clear_drift()

        for p in self.loaders:
            p.play()
            self.players.append(p)
        self.loaders.clear()

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

        sync_frames:Dict[FrameType, ndarray] = {}
        with self.sync_lock:
            cam_frames: Dict[int, Dict[FrameType, ndarray]] = self.frame_sync_dict[cam_id]
            if frame_id not in cam_frames:
                cam_frames[frame_id] = {}
            cam_frames[frame_id][frame_type] = frame
            for t in self.types:
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

        for ft in sync_frames.keys():
            for callback in self.frameCallbacks:
                callback(cam_id, ft, sync_frames[ft])

    def _clear_frame_sync(self) -> None:
        with self.sync_lock:
            for i in range(self.num_cams):
                self.frame_sync_dict[i].clear()

    # DRIFT CALCULATION
    def _calcuate_drift(self, cam_id, frame_id: int) -> None:
        drift_ids: Dict[int, int] = {}
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
    def play(self, value: bool, name: str = '') -> None:
        if value:
            if not name in self.folders:
                logger.warning(f"Folder {name} not found")
                return
            message: Message = Message(MessageType.START, name)
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

    def clear_state_messages(self) -> None:
        with self.state_messages.mutex:
            self.state_messages.queue.clear()

    def get_num_chunks(self) -> int:
        return self.num_chunks

    def get_drift(self) -> int:
        with self.drift_lock:
            return self.drift

    # CALLBACKS
    def addFrameCallback(self, callback: FrameCallback) -> None:
        if self.running:
            logger.warning('Camera: cannot add callback while player is running')
            return
        self.frameCallbacks.add(callback)

    # STATIC METHODS
    @staticmethod
    def _get_video_folders(settings: SimulatorSettings, data_path: str = "") -> FolderDict:
        folders: FolderDict = {}
        video_path: Path = Path(data_path) / settings.video_path
        if not video_path.is_dir():
            return folders
        for folder in video_path.iterdir():
            if folder.is_dir():
                max_chunk: int = -1
                for file in folder.iterdir():
                    if file.is_file() and (file.name.endswith(CoderFormat.H264.value) or file.name.endswith(CoderFormat.H265.value)):
                        n: str = file.stem.split('_')[2]
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
            self._prepare_folder(folder)
            self.play(True, folder)

    def _on_refresh_path(self, _=None) -> None:
        self.folders = self._get_video_folders(self.settings, self.data_path)
        self.settings.available_folders = list(self.folders.keys())

    def _on_range_changed(self, _=None) -> None:
        self._recompute_range()
        if self._suppress_restart:
            return
        # Restart playback so the new range takes effect immediately.
        if self.active_folder and self.active_folder in self.folders:
            self.clear_state_messages()
            self.play(True, self.active_folder)

    def update_gui(self, frame_id: int = 0) -> None:
        chunk: int = self.get_current_chunk()
        self.settings.current_chunk = chunk
        self.settings.playback_time = self._format_time(self._global_frame(chunk, frame_id) / self.fps)

    def _global_frame(self, chunk: int, frame_id: int) -> int:
        """Absolute frame index across the whole timeline for (chunk, in-chunk frame)."""
        if chunk < 0:
            return 0
        with self.playback_lock:
            chunk_frames: list[int] = self.chunk_frames
        return sum(chunk_frames[:chunk]) + frame_id

    # RANGE / TIMELINE
    def _prepare_folder(self, folder: str) -> None:
        """Probe every chunk's frame count and reset the range to the full timeline."""
        chunk_frames: list[int] = self._probe_chunk_frames(folder)
        with self.playback_lock:
            self.chunk_frames = chunk_frames
        self.settings.max_chunks = len(chunk_frames)
        # Reset norms without triggering a restart; the caller starts playback.
        self._suppress_restart = True
        try:
            self.settings.start_norm = 0.0
            self.settings.end_norm = 1.0
        finally:
            self._suppress_restart = False
        self._recompute_range()

    def _probe_chunk_frames(self, folder: str) -> list[int]:
        """Frame count of each chunk (cam 0 / first frame type), indexed 0..max_chunk."""
        if folder not in self.folders:
            return []
        folder_obj: Folder = self.folders[folder]
        frame_type: FrameType = self.types[0]
        frames: list[int] = []
        for chunk in range(folder_obj.chunks + 1):
            path: Path = folder_obj.path / make_file_name(0, frame_type, chunk, self.suffix)
            count: int = StreamReader.frame_count(str(path)) if path.is_file() else 0
            frames.append(count)
        return frames

    def _recompute_range(self) -> None:
        """Map normalized start/end onto global frame positions and chunk offsets."""
        with self.playback_lock:
            chunk_frames: list[int] = self.chunk_frames
            total: int = sum(chunk_frames)

        if total <= 0:
            with self.playback_lock:
                self.start_chunk = self.start_offset = 0
                self.end_chunk = self.end_offset = 0
            self.settings.start_time = self._format_time(0.0)
            self.settings.end_time = self._format_time(0.0)
            return

        a: float = max(0.0, min(1.0, self.settings.start_norm))
        b: float = max(0.0, min(1.0, self.settings.end_norm))
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

        self.settings.start_time = self._format_time(start_global / self.fps)
        self.settings.end_time = self._format_time(end_global / self.fps)

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

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes: int = int(seconds // 60)
        secs: float = seconds - minutes * 60
        return f"{minutes}:{secs:04.1f}"