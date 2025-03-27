from threading import Thread, Event
from pathlib import Path
from numpy import ndarray
from typing import Set, Dict
from enum import Enum, auto
from queue import Queue

from modules.cam.DepthAi.Definitions import FrameType, FrameCallback
from modules.cam.player.Player import Player, DecoderType
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

class SyncPlayer(Thread):
    def __init__(self, input_path: str, num_cams: int, types: list[FrameType], decoder: DecoderType) -> None:
        super().__init__()
        self.input_path: Path = Path(input_path)
        self.num_cams: int = num_cams
        self.types: list[FrameType] = types

        self.state_messages: Queue[StateMessage] = Queue()

        self.stop_event = Event()

        self.playback_path: Path = Path()
        self.chunk: int = -1

        self.folders: Dict[Path, int] = self._get_video_folders(self.input_path)

        self.players: Dict[int, Dict[FrameType, Player]] = {
            c: {t: Player(c, t, self._frame_callback, self._stop_callback, decoder) for t in self.types}
            for c in range(self.num_cams)
        }

        self.frameCallbacks: Dict[FrameType, Set[FrameCallback]] = {t: set() for t in self.types}

    def stop(self) -> None:
        self.play(False)
        self.stop_event.set()
        self.join()

    def run(self) -> None:
        while not self.stop_event.is_set() and self.state_messages.empty():
            state_message: StateMessage = self.state_messages.get()

            if state_message.state == State.PLAY:
                if type(state_message.value) is str:
                    self.chunk = 0
                    self.playback_path = Path(state_message.value)
                    self._start_players()
            elif state_message.state == State.STOP:
                self.chunk = -1
                self._stop_players()
            elif state_message.state == State.NEXT:
                if type(state_message.value) is int:
                    chunk: int = state_message.value
                    if self.chunk == chunk:
                        max_chunk: int = self.folders[self.playback_path]
                        self.chunk = (self.chunk + 1) % (max_chunk + 1)
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
        for callback in self.frameCallbacks[frameType]:
            callback(cam_id, frameType, frame)

    def _stop_callback(self, chunk_id: int) -> None:
        message: StateMessage = StateMessage(State.NEXT, chunk_id)
        self.state_messages.put(message)

    # EXTERNAL METHODS
    def play(self, value: bool, path: str = '') -> None:
        if value:
            if not path in self.folders:
                print(f"Folder {path} not found")
                return
            message: StateMessage = StateMessage(State.PLAY, path)
        else:
            message: StateMessage = StateMessage(State.STOP)
        self.state_messages.put(message)

    def get_folders(self) -> list[str]:
        return [str(f) for f in self.folders.keys()]

    def get_chunks(self, folder: str) -> int:
        return self.folders.get(Path(folder), 0)

    # CALLBACKS
    def addFrameCallback(self, frameType: FrameType, callback: FrameCallback) -> None:
        self.frameCallbacks[frameType].add(callback)
    def discardFrameCallback(self, frameType: FrameType, callback: FrameCallback) -> None:
        self.frameCallbacks[frameType].discard(callback)
    def clearFrameCallbacks(self) -> None:
        self.frameCallbacks.clear()

    # STATIC METHODS
    @staticmethod
    def _get_video_folders(path: Path) -> Dict[Path, int]:
        folders: Dict[Path, int] = {}
        for folder in path.iterdir():
            if folder.is_dir():
                max_chunk: int = max(
                    (int(file.name.split('_')[2]) for file in folder.iterdir() if file.is_file() and file.name.endswith('.mp4') and file.name.split('_')[2].isdigit()),
                    default=0
                )
                if max_chunk > 0:
                    folders[folder] = max_chunk
        return folders