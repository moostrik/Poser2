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

        self.playback_name: str = ''
        self.playback_chunk: int = -1

        self.folders: FolderDict = self._get_video_folders(settings)
        self.current_folder: str = ''

        self.hwt: str = HwaccelString[settings.decoder]
        self.hwd: str = HwaccelDeviceString[settings.decoder]

        self.players: list[FFmpegPlayer] = []
        self.loaders: list[FFmpegPlayer] = []
        self.closers: list[FFmpegPlayer] = []

        self.callback_lock: Lock = Lock()
        self.frameCallbacks: Set[FrameCallback] = set()

    def stop(self) -> None:
        self.clearFrameCallbacks()
        self.play(False)
        self.stop_event.set()
        self.join()

    def run(self) -> None:
        state: State = State.IDLE

        while True:
            message: Message | None = None
            try:
                message = self.state_messages.get(block=False)
            except Exception as e:
                message = None

            if message:
                print(f"Message: {message.state}, {message.value}")

            if message and message.state == MessageType.START:
                # make thread safe
                    self.current_folder = message.value
                    self.playback_chunk = -1
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
            self.clean()

            if self.stop_event.is_set() and state == State.IDLE:
                break

            # print(state)
            sleep(0.01)

    def _load(self) -> None:
        folder: Folder = self.folders[self.current_folder]
        self.playback_chunk = (self.playback_chunk + 1) % (folder.chunks + 1)

        for c in range(self.num_cams):
            for t in self.types:
                path: Path = folder.path / make_file_name(c, t, self.playback_chunk)
                if path.is_file():
                    player: FFmpegPlayer = FFmpegPlayer(c, t, self._frame_callback, self.hwt, self.hwd)
                    player.load(str(path), self.playback_chunk)
                    self.loaders.append(player)

    def _start(self) -> None:
        for p in self.loaders:
            p.play()
            self.players.append(p)
        self.loaders.clear()

    def _stop(self) -> None:
        for p in self.loaders:
            p.stop()
            self.closers.append(p)
        self.players.clear()
        for p in self.players:
            p.stop()
            self.closers.append(p)
        self.players.clear()

    def clean(self) -> None:
        for p in self.closers:
            if not p.is_playing() and not p.is_loading():
                self.closers.remove(p)

    def _finished_loading(self) -> bool:
        for p in self.loaders:
            if not p.is_loaded():
                return False
        return True

    def _finished_playing(self) -> bool:
        for p in self.players:
            if not p.is_playing():
                return True
        return False

    def _finished_stopping(self) -> bool:
        for p in self.closers:
            if p.is_playing() or p.is_loading():

                return False
        return True

    def _frame_callback(self, cam_id: int, frameType: FrameType, frame: ndarray) -> None:
        with self.callback_lock:
            for callback in self.frameCallbacks:
                callback(cam_id, frameType, frame)

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
        return self.playback_chunk

    # CALLBACKS
    def addFrameCallback(self, callback: FrameCallback) -> None:
        with self.callback_lock:
            self.frameCallbacks.add(callback)
    def discardFrameCallback(self, callback: FrameCallback) -> None:
        with self.callback_lock:
            self.frameCallbacks.discard(callback)
    def clearFrameCallbacks(self) -> None:
        with self.callback_lock:
            self.frameCallbacks.clear()

    # STATIC METHODS
    @staticmethod
    def _get_video_folders(settings: Settings) -> FolderDict :
        folders: FolderDict = {}
        video_path: Path = Path(settings.video_path)
        for folder in video_path.iterdir():
            if folder.is_dir():
                if not is_folder_for_settings(str(folder), settings):
                    continue
                max_chunk: int = 0
                for file in folder.iterdir():
                    if file.is_file() and file.name.endswith('.mp4'):
                        n: str = file.stem.split('_')[2]
                        if n.isdigit():
                            max_chunk = max(max_chunk, int(n))
                if max_chunk > 0:
                    folders[folder.name] = (Folder(folder.name, folder, max_chunk))
        return folders

