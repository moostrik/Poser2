from threading import Thread, Event
from pathlib import Path
from modules.cam.DepthAi.Definitions import FrameType, FrameTypeString
from modules.cam.player.Player import Player, DecoderType

class SyncPlayer(Thread):
    def __init__(self, input_path: str, num_cams: int, types: list[FrameType], decoder: DecoderType) -> None:
        super().__init__()
        self.input_path: Path = Path(input_path)
        self.num_cams: int = num_cams
        self.types: list[FrameType] = types
        self.running = False
        self.stop_event = Event()

        self.players: dict[int, dict[FrameType, Player]] = {}
        for c in range(self.num_cams):
            self.players[c] = {}
            for t in self.types:
                self.players[c][t] = Player(decoder)

    def run(self) -> None:
        self.running = True
        self._start_players()

        while self.running:
            if self.stop_event.wait(timeout=0.01):
                self.stop_event.clear()
                self.running = False
                break

        self._stop_players()

    def stop(self) -> None:
        self.stop_event.set()
        self.join()

    def _start_players(self) -> None:
        for c in range(self.num_cams):
            for t in self.types:
                player: Player | None = self.players[c].get(t)
                if player:
                    id: str = f"{c}_{FrameTypeString[t]}"
                    path: Path = self.input_path / f"{c}_{FrameTypeString[t]}_000.mp4"
                    player.start(c, t, str(path), self._callback)

    def _stop_players(self) -> None:
        for c in range(self.num_cams):
            for t in self.types:
                player: Player | None = self.players[c].get(t)
                if player:
                    player.stop()

    def _callback(self, cam_id: str, frameType: FrameType, frame) -> None:
        pass