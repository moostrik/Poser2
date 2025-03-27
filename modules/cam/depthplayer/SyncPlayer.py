from threading import Thread, Event
from pathlib import Path
from numpy import ndarray
from typing import Set, Dict
from enum import Enum, auto
from queue import Queue

from modules.cam.depthcam.Definitions import FrameType, FrameCallback
from modules.cam.depthplayer.Player import Player, DecoderType
from modules.cam.recorder.SyncRecorder import make_path

class State(Enum):
    IDLE = auto()
    PLAY = auto()
    STOP = auto()
    NEXT = auto()

class StateMessage():
    def __init__(self, state: State, value = None) -> None:
        self.state: State = state
        self.value = value

class Folder():
    def __init__(self, name: str, path: str, chunks: int) -> None:
        self.name: str = name
        self.path: str = path
        self.chunks: int = chunks

FolderDict = Dict[str, Folder]

class SyncPlayer(Thread):
    def __init__(self, input_path: str, num_cams: int, types: list[FrameType], decoder: DecoderType) -> None:
        super().__init__()
        self.input_path: Path = Path(input_path)
        self.num_cams: int = num_cams
        self.types: list[FrameType] = types

        self.state_messages: Queue[StateMessage] = Queue()

        self.stop_event = Event()

        self.playback_path: Path = Path()
        self.playback_name: str = ''
        self.chunk: int = -1
        self.max_chunk: int = -1

        self.folders: FolderDict = self._get_video_folders(self.input_path)

        self.players: Dict[int, Dict[FrameType, Player]] = {
            c: {t: Player(c, t, self._frame_callback, self._stop_callback, decoder) for t in self.types}
            for c in range(self.num_cams)
        }

        self.frameCallbacks: Set[FrameCallback] = set()

    def stop(self) -> None:
        self.play(False)
        self.stop_event.set()
        self.join()

    def run(self) -> None:
        while not self.stop_event.is_set() and self.state_messages.empty():
            state_message: StateMessage = self.state_messages.get()
            print(state_message.state)
            if state_message.state == State.PLAY:
                if type(state_message.value) is str:
                    name: str = state_message.value
                    self.chunk = 0
                    self.max_chunk = self.folders[name].chunks
                    self.playback_path = Path(self.folders[name].path)
                    self._start_players()
                    print(f"Playing {name}")
            elif state_message.state == State.STOP:
                self.chunk = -1
                self._stop_players()
            elif state_message.state == State.NEXT:
                if type(state_message.value) is int:
                    chunk: int = state_message.value
                    if self.chunk == chunk:
                        self.chunk = (self.chunk + 1) % (self.max_chunk + 1)
                        print(f"Chunk {self.chunk}")
                        self._stop_players()
                        self._start_players()

    def _start_players(self) -> None:
        for c in range(self.num_cams):
            for t in self.types:
                player: Player | None = self.players[c].get(t)
                if player:
                    path: Path = make_path(self.playback_path, c, t, self.chunk)
                    if path.is_file():
                        player.start(str(path), self.chunk)
                    else:
                        print(f"File {path} not found")

    def _stop_players(self) -> None:
        for c in range(self.num_cams):
            for t in self.types:
                player: Player | None = self.players[c].get(t)
                if player:
                    player.stop()

    def _frame_callback(self, cam_id: int, frameType: FrameType, frame: ndarray) -> None:
        for callback in self.frameCallbacks:
            callback(cam_id, frameType, frame)

    def _stop_callback(self, chunk_id: int) -> None:
        message: StateMessage = StateMessage(State.NEXT, chunk_id)
        self.state_messages.put(message)

    # EXTERNAL METHODS
    def play(self, value: bool, name: str = '') -> None:
        if value:
            if not name in self.folders:
                print(f"Folder {name} not found")
                return
            message: StateMessage = StateMessage(State.PLAY, name)
        else:
            message: StateMessage = StateMessage(State.STOP)
        print('sending message', message.state)
        self.state_messages.put(message)

    def get_folders(self) -> list[str]:
        return list(self.folders.keys())

    # def get_chunks(self, folder: str) -> int:
    #     return self.folders.get(Path(folder), 0)

    # CALLBACKS
    def addFrameCallback(self, callback: FrameCallback) -> None:
        self.frameCallbacks.add(callback)
    def discardFrameCallback(self, callback: FrameCallback) -> None:
        self.frameCallbacks.discard(callback)
    def clearFrameCallbacks(self) -> None:
        self.frameCallbacks.clear()

    # STATIC METHODS
    @staticmethod
    def _get_video_folders(path: Path) -> FolderDict :
        folders: FolderDict = {}
        for paths in path.iterdir():
            if paths.is_dir():
                max_chunk: int = 0
                for file in paths.iterdir():
                    if file.is_file() and file.name.endswith('.mp4'):
                        n: str = file.stem.split('_')[2]
                        if n.isdigit():
                            max_chunk = max(max_chunk, int(n))
                if max_chunk > 0:
                    folders[paths.name] = (Folder(paths.name, str(paths), max_chunk))
        return folders