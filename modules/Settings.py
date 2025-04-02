from __future__ import annotations
from enum import Enum



class Settings():
    class CoderType(Enum):
        CPU =   0
        GPU =   1
        iGPU =  2

    def __init__(self) -> None:
        # PATHS
        self.root_path: str
        self.model_path: str
        self.video_path: str
        self.temp_path: str
        self.file_path: str

        # CAMERA SETTINGS
        self.camera_list: list[str]
        self.fps: int
        self.color: bool
        self.stereo: bool
        self.person: bool
        self.lowres: bool
        self.show_stereo: bool
        self.simulation: bool
        self.passthrough: bool

        # DETECTION SETTINGS
        self.num_players: int
        self.lightning: bool
        self.pose: bool

        # RECORDER AND PLAYER SETTINGS
        self.chunk_length: float
        self.encoder: Settings.CoderType
        self.decoder: Settings.CoderType

    def check(self) -> None:
        for key, value in vars(self).items():
            if value is None:
                raise ValueError(f"'{key}' is not set")