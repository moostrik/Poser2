from threading import Thread, Event
from pathlib import Path
import numpy as np
from typing import Set, Callable, Dict
import time
from enum import Enum, auto

from modules.cam.DepthAi.Definitions import FrameType, FrameCallback
from modules.cam.player.Player import Player, DecoderType
from modules.cam.recorder.SyncRecorder import make_path

class PlayerState(Enum):
    IDLE = auto()
    PLAYING = auto()
    STOPPED = auto()
    NEXT_CHUNK = auto()

class SyncPlayer(Thread):
    def __init__(self, input_path: str, num_cams: int, types: list[FrameType], decoder: DecoderType) -> None:
        super().__init__()
        self.input_path: Path = Path(input_path)
        self.num_cams: int = num_cams
        self.types: list[FrameType] = types

        self.state: PlayerState = PlayerState.IDLE
        self.state_event = Event()
        self.stop_event = Event()

        self.playback_path: Path = Path()
        self.chunk: int = 0

        self.folders: Dict[Path, int] = self._get_video_folders(self.input_path)

        self.players: Dict[int, Dict[FrameType, Player]] = {
            c: {t: Player(c, t, self._frame_callback, self._stop_callback, decoder) for t in self.types}
            for c in range(self.num_cams)
        }

        self.frameCallbacks: Dict[FrameType, Set[FrameCallback]] = {t: set() for t in self.types}

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.state_event.wait()
            self.state_event.clear()

            if self.state == PlayerState.PLAYING:
                self._start_players()
            elif self.state == PlayerState.STOPPED:
                self._stop_players()
                self.state = PlayerState.IDLE
            elif self.state == PlayerState.NEXT_CHUNK:
                self._stop_players()
                self.chunk = (self.chunk + 1) % (max(self.folders.values()) + 1)
                self._start_players()
                self.state = PlayerState.PLAYING

            time.sleep(0.01)

    def stop(self) -> None:
        self.stop_playback()
        self.stop_event.set()
        self.join()

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

    def _frame_callback(self, cam_id: int, frameType: FrameType, frame: np.ndarray) -> None:
        for callback in self.frameCallbacks[frameType]:
            callback(cam_id, frameType, frame)

    def _stop_callback(self, chunk_id: int) -> None:
        if chunk_id == self.chunk:
            self.state = PlayerState.NEXT_CHUNK
            self.state_event.set()

    # EXTERNAL METHODS
    def start_playback(self, path: str) -> None:
        if self.state == PlayerState.PLAYING:
            print('Already playing')
            return

        if Path(path) not in self.folders:
            print(f"Folder {path} not found")
            return

        self.chunk = 0
        self.playback_path = Path(path)
        self.state = PlayerState.PLAYING
        self.state_event.set()

    def stop_playback(self) -> None:
        if self.state != PlayerState.PLAYING:
            print('Not playing')
            return

        self.state = PlayerState.STOPPED
        self.state_event.set()

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