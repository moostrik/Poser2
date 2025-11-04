from threading import Thread, Event, Lock
from pathlib import Path
from numpy import ndarray
from typing import Set, Dict
from enum import Enum, auto
from queue import Queue
from time import sleep

from modules.Settings import Settings
from modules.cam.depthcam.Definitions import FrameType, FrameCallback
from modules.cam.depthplayer.FFmpegPlayer import FFmpegPlayer
from modules.cam.recorder.SyncRecorder import make_file_name, is_folder_for_settings

class State(Enum):
    IDLE = auto()
    LOAD = auto()
    LOADING = auto()
    PLAYING = auto()
    STOP = auto()
    STOPPING = auto()
    NEXT = auto()

HwaccelString: dict[Settings.CoderType, str] = {
    Settings.CoderType.CPU:  '',
    Settings.CoderType.GPU:  'd3d12va',
    Settings.CoderType.iGPU: 'd3d12va'
}

HwaccelDeviceString: dict[Settings.CoderType, str] = {
    Settings.CoderType.CPU:  '',
    Settings.CoderType.GPU:  '0',
    Settings.CoderType.iGPU: '1'
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

class SyncPlayer(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.input_path: Path = Path(settings.path_video)
        self.num_cams: int = settings.camera_num
        self.types: list[FrameType] = settings.video_frame_types
        self.fps: float = settings.camera_player_fps

        self.running: bool = False
        self.state_messages: Queue[Message] = Queue()

        self.stop_event = Event()

        self.play_chunk: int = -1
        self.load_chunk: int = -1
        self.num_chunks: int = 0
        self.chunk_range_0: int = 0
        self.chunk_range_1: int = 0
        self.load_folder: str = ''
        self.suffix: str = settings.video_format.value

        self.folders: FolderDict = self._get_video_folders(settings)

        self.hwt: str = HwaccelString[settings.video_decoder]
        self.hwd: str = HwaccelDeviceString[settings.video_decoder]

        self.players: list[FFmpegPlayer] = []
        self.loaders: list[FFmpegPlayer] = []
        self.closers: list[FFmpegPlayer] = []

        self.sync_lock: Lock = Lock()
        self.frame_sync_dict: Dict[int, Dict[int, Dict[FrameType, ndarray]]] = {}
        for i in range(self.num_cams):
            self.frame_sync_dict[i] = {}

        self.drift_lock: Lock = Lock()
        self.drift_ids: Dict[int, int] = {}
        self.drift:int = 0

        self.playback_lock: Lock = Lock()
        self.frameCallbacks: Set[FrameCallback] = set()

    def stop(self) -> None:
        self._set_play_chunk(-1)
        message: Message = Message(MessageType.STOP)
        self.state_messages.put(message)
        self.stop_event.set()
        self.join()

    def run(self) -> None:
        state: State = State.IDLE
        self.running = True

        while self.running:
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

            sleep(0.01)

    def _load(self) -> None:
        folder: Folder = self.folders[self.load_folder]
        LC: int = self._get_load_chunk() + 1
        LC = max(LC, self.chunk_range_0)
        LC = min(LC, self.chunk_range_1 + 1)
        if LC > self.chunk_range_1:
            LC = self.chunk_range_0

        self._set_load_chunk(LC)

        for c in range(self.num_cams):
            for t in self.types:
                path: Path = folder.path / make_file_name(c, t, self.load_chunk, self.suffix)
                if path.is_file():

                    player: FFmpegPlayer = FFmpegPlayer(c, t, self._frame_sync_callback, self.hwt, self.hwd, self.fps)
                    player.load(str(path), self.load_chunk)
                    self.loaders.append(player)
                else:
                    print(f"File {path} not found")

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
            self.update_gui()

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
                    print(f"Frame {key} is older than {frame_id}, marking for removal")
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
                print(f"Folder {name} not found")
                return
            message: Message = Message(MessageType.START, name)
        else:
            message: Message = Message(MessageType.STOP)

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

    def set_chunk_range(self, R0: int, R1: int) -> None:
        with self.playback_lock:
            self.chunk_range_0 = R0
            self.chunk_range_1 = R1

    def get_drift(self) -> int:
        with self.drift_lock:
            return self.drift

    # CALLBACKS
    def addFrameCallback(self, callback: FrameCallback) -> None:
        if self.running:
            print('Camera: cannot add callback while player is running')
            return
        self.frameCallbacks.add(callback)

    # STATIC METHODS
    @staticmethod
    def _get_video_folders(settings: Settings) -> FolderDict :
        folders: FolderDict = {}
        suffix: str = settings.video_format.value
        video_path: Path = Path(settings.path_video)
        for folder in video_path.iterdir():
            if folder.is_dir():
                if not is_folder_for_settings(str(folder), settings):
                    continue
                max_chunk: int = -1
                for file in folder.iterdir():
                    if file.is_file() and (file.name.endswith(Settings.CoderFormat.H264.value) or file.name.endswith(Settings.CoderFormat.H265.value)):
                        n: str = file.stem.split('_')[2]
                        if n.isdigit():
                            max_chunk = max(max_chunk, int(n))
                if max_chunk >= 0:
                    folders[folder.name] = (Folder(folder.name, folder, max_chunk))
        return folders

    # GUI HACK
    def update_gui(self) -> None:
        pass