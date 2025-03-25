from threading import Thread, Event
from pathlib import Path
import numpy as np
from pathlib import Path
from typing import Set
import time

from modules.cam.DepthAi.Definitions import FrameType, FrameTypeString, FrameCallback
from modules.cam.player.Player import Player, DecoderType
from modules.cam.recorder.SyncRecorder import make_path

class SyncPlayer(Thread):
    def __init__(self, input_path: str, num_cams: int, types: list[FrameType], decoder: DecoderType) -> None:
        super().__init__()
        self.input_path: Path = Path(input_path)
        self.num_cams: int = num_cams
        self.types: list[FrameType] = types
        self.running: bool = False
        self.playing: bool = False
        self.playback_path: Path = Path()
        self.chunk: int = 0

        self.start_playback_event = Event()
        self.stop_playback_event = Event()
        self.next_playback_event = Event()
        self.stop_event = Event()

        self.folders: dict[Path, int] = self._get_video_folders(self.input_path)

        self.players: dict[int, dict[FrameType, Player]] = {}
        for c in range(self.num_cams):
            self.players[c] = {}
            for t in self.types:
                self.players[c][t] = Player(c, t, self._frame_callback, self._stop_callback, decoder)

        self.frameCallbacks: dict[FrameType, Set[FrameCallback]] = {}

    def run(self) -> None:
        self.running = True

        while self.running:
            if self.stop_event.is_set():
                self.stop_playback_event.set()
                self.running = False
            if self.start_playback_event.is_set():
                self.start_playback_event.clear()
                self._start_players()
            if self.next_playback_event.is_set():
                self.next_playback_event.clear()
                self._next_players()
            if self.stop_playback_event.is_set():
                self.stop_playback_event.clear()
                self._stop_players()
            time.sleep(0.01)

    def stop(self) -> None:
        self.stop_event.set()
        self.join()

    def _start_players(self) -> None:
        for c in range(self.num_cams):
            for t in self.types:
                player: Player | None = self.players[c].get(t)
                if player:
                    path: Path = make_path(self.playback_path, c, t, self.chunk)
                    player.start(str(path), self.chunk)

    def _stop_players(self) -> None:
        for c in range(self.num_cams):
            for t in self.types:
                player: Player | None = self.players[c].get(t)
                if player:
                    player.stop()

    def _next_players(self) -> None:
        self._stop_players()
        self._start_players()

    def _frame_callback(self, cam_id: int, frameType: FrameType, frame: np.ndarray) -> None:
        for c in self.frameCallbacks[frameType]:
            c(cam_id, frameType, frame)

    def _stop_callback(self, chunk_id: int) -> None:
        if chunk_id == self.chunk:
            self.chunk += 1
            self.next_playback_event.set()
        pass

    # EXTERNAL METHODS
    def start_playback(self, path: str) -> None:
        if self.playing:
            print('Already playing')
            return

        if Path(path) not in self.folders:
            print(f"Folder {path} not found")
            return

        self.chunk = 0
        self.playing = True
        self.playback_path = Path(path)
        self.start_playback_event.set()

    def stop_playback(self) -> None:
        if not self.playing:
            print('Not playing')
            return

        self.playing = False
        self.stop_playback_event.set()

    def get_folders(self) -> list[str]:
        return [str(f) for f in self.folders.keys()]

    def get_chunks(self, folder: str) -> int:
        if folder in self.folders:
            return self.folders[Path(folder)]
        return 0

    # CALLBACKS
    def addFrameCallback(self, frameType: FrameType, callback: FrameCallback) -> None:
        self.frameCallbacks[frameType].add(callback)
    def discardFrameCallback(self, frameType: FrameType, callback: FrameCallback) -> None:
        self.frameCallbacks[frameType].discard(callback)
    def clearFrameCallbacks(self) -> None:
        self.frameCallbacks.clear()

    # STATIC METHODS
    @staticmethod
    def _get_video_folders(path: Path) -> dict[Path, int]:
        folders: dict[Path, int] = {}
        chuncks: int = 0
        for folder in path.iterdir():
            if folder.is_dir():
                files = list(folder.iterdir())
                for file in files:
                    if file.is_file():
                        if file.name.endswith('.mp4'):
                            segments: list[str] = file.name.split('_')
                            if len(segments) == 3:
                                if segments[2].isdigit():
                                    d = int(segments[2])
                                    if d > chuncks:
                                        chuncks = d
                if chuncks > 0:
                    folders[folder] = chuncks
        return folders