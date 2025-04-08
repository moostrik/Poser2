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
        self.input_path: Path = Path(settings.video_path)
        self.num_cams: int = settings.num_cams
        self.types: list[FrameType] = settings.frame_types

        self.state_messages: Queue[Message] = Queue()

        self.stop_event = Event()

        self.load_chunk: int = -1
        self.load_folder: str = ''
        self.suffix: str = settings.format.value

        self.folders: FolderDict = self._get_video_folders(settings)

        self.hwt: str = HwaccelString[settings.decoder]
        self.hwd: str = HwaccelDeviceString[settings.decoder]

        self.players: list[FFmpegPlayer] = []
        self.loaders: list[FFmpegPlayer] = []
        self.closers: list[FFmpegPlayer] = []

        self.sync_lock: Lock = Lock()
        self.frame_sync_dict: Dict[int, Dict[int, Dict[FrameType, ndarray]]] = {}
        for i in range(self.num_cams):
            self.frame_sync_dict[i] = {}

        self.playback_lock: Lock = Lock()
        self.frameCallbacks: Set[FrameCallback] = set()

    def stop(self) -> None:
        self.play(False)
        self.stop_event.set()

    def run(self) -> None:
        state: State = State.IDLE

        while True:
            message: Message | None = None
            try:
                message = self.state_messages.get(block=False)
            except Exception as e:
                message = None

            if message and message.state == MessageType.START:
                self._set_load_folder(message.value)
                self._set_load_chunk(-1)
                state = State.LOAD
            if message and message.state == MessageType.STOP:
                state = State.STOP

            if state == State.LOAD:
                self._stop()
                self._load()
                state = State.LOADING
            if state == State.LOADING and self._finished_loading():
                self._start()
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
                break

            sleep(0.01)

    def _load(self) -> None:
        folder: Folder = self.folders[self.load_folder]
        self._set_load_chunk((self._get_load_chunk() + 1) % (folder.chunks + 1))

        for c in range(self.num_cams):
            for t in self.types:
                path: Path = folder.path / make_file_name(c, t, self.load_chunk, self.suffix)
                if path.is_file():

                    player: FFmpegPlayer = FFmpegPlayer(c, t, self._frame_callback, self.hwt, self.hwd)
                    player.load(str(path), self.load_chunk)
                    self.loaders.append(player)
                else:
                    print(f"File {path} not found")

    def _start(self) -> None:
        for p in self.players:
            p.stop()
            self.closers.append(p)
        self.players.clear()
        for p in self.loaders:
            p.play()
            self.players.append(p)
        self.loaders.clear()

    def _stop(self) -> None:
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

    def _set_load_chunk(self, value: int) -> None:
        with self.playback_lock:
            self.load_chunk = value

    def _get_load_chunk(self) -> int:
        with self.playback_lock:
            return self.load_chunk

    def _set_load_folder(self, value: str) -> None:
        with self.playback_lock:
            self.load_folder = value

    def _get_load_folder(self) -> str:
        with self.playback_lock:
            return self.load_folder

    def _frame_callback(self, cam_id: int, frame_type: FrameType, frame: ndarray, frame_id) -> None:
        sync_frames:Dict[FrameType, ndarray] = {}
        with self.sync_lock:
            cam_frames: Dict[int, Dict[FrameType, ndarray]] = self.frame_sync_dict[cam_id]
            if frame_id not in cam_frames:
                cam_frames[frame_id] = {}
            cam_frames[frame_id][frame_type] = frame
            for t in self.types:
                if t not in cam_frames[frame_id]:
                    return

            for key in cam_frames.keys():
                if key < frame_id:
                    print(f"Frame {key} is older than {frame_id}, removing")
                    del cam_frames[key]

            sync_frames = cam_frames[key]
            del cam_frames[frame_id]

        print(cam_id, frame_id)
        for ft in sync_frames.keys():
            for callback in self.frameCallbacks:
                callback(cam_id, ft, sync_frames[ft], frame_id)


    # EXTERNAL METHODS
    def play(self, value: bool, name: str = '') -> None:
        if value:
            if not name in self.folders:
                print(f"Folder {name} not found")
                return
            message: Message = Message(MessageType.START, name)
        else:
            message: Message = Message(MessageType.STOP)
        self.state_messages.put(message)

    def get_folder_names(self) -> list[str]:
        return list(self.folders.keys())

    def get_chunks(self, folder: str) -> int:
        return self.folders[folder].chunks if folder in self.folders else 0

    def get_current_chunk(self) -> int:
        return max(self._get_load_chunk() - 1, 0)

    def get_current_folder(self) -> str:
        return self._get_load_folder()

    def clear_state_messages(self) -> None:
        with self.state_messages.mutex:
            self.state_messages.queue.clear()

    # CALLBACKS
    def addFrameCallback(self, callback: FrameCallback) -> None:
        self.frameCallbacks.add(callback)
    def discardFrameCallback(self, callback: FrameCallback) -> None:
        self.frameCallbacks.discard(callback)
    def clearFrameCallbacks(self) -> None:
        self.frameCallbacks.clear()

    # STATIC METHODS
    @staticmethod
    def _get_video_folders(settings: Settings) -> FolderDict :
        folders: FolderDict = {}
        suffix: str = settings.format.value
        video_path: Path = Path(settings.video_path)
        for folder in video_path.iterdir():
            if folder.is_dir():
                if not is_folder_for_settings(str(folder), settings):
                    continue
                max_chunk: int = 0
                for file in folder.iterdir():
                    if file.is_file() and (file.name.endswith(Settings.CoderFormat.H264.value) or file.name.endswith(Settings.CoderFormat.H265.value)):
                        n: str = file.stem.split('_')[2]
                        if n.isdigit():
                            max_chunk = max(max_chunk, int(n))
                if max_chunk > 0:
                    folders[folder.name] = (Folder(folder.name, folder, max_chunk))
        return folders

