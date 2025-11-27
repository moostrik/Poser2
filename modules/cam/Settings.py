from dataclasses import dataclass, field
from enum import Enum

from modules.cam.depthcam.Definitions import FrameType

@dataclass
class Settings():
    class CoderType(Enum):
        CPU =   0
        GPU =   1
        iGPU =  2

    class CoderFormat(Enum):
        H264 = '.mp4'
        H265 = '.hevc'

    # CAMERA SETTINGS
    ids: list[str]          = field(default_factory=list)
    num: int                = field(default=0, init=False)
    fps: float              = field(default=30.0)
    yolo: bool              = field(default=True)
    color: bool             = field(default=True)
    square: bool            = field(default=True)
    stereo: bool            = field(default=False)
    hd_ready: bool          = field(default=False)
    show_stereo: bool       = field(default=False)
    manual: bool            = field(default=False)
    flip_h: bool            = field(default=False)
    flip_v: bool            = field(default=False)
    perspective: float      = field(default=0.0)

    model_path: str         = field(default="models")
    video_path: str         = field(default="recordings")
    temp_path: str          = field(default="temp")

    sim_enabled: bool       = field(default=False)
    sim_passthrough: bool   = field(default=False)
    sim_fps: float          = field(default=30.0)

    rec_enabled: bool       = field(default=False)

    video_chunk_length: float = field(default=10.0) # in seconds
    video_encoder: CoderType  = field(default=CoderType.iGPU)
    video_decoder: CoderType  = field(default=CoderType.iGPU)
    video_format: CoderFormat = field(default=CoderFormat.H264)
    video_frame_types: list[FrameType] = field(default_factory=lambda: [FrameType.VIDEO])

    def __post_init__(self) -> None:
        self.num = len(self.ids)
        if self.sim_enabled:
            if self.rec_enabled:
                print("Cam Settings: Warning - Both simulation and recording are enabled, disabling recording.")
                self.rec_enabled = False
            if self.sim_fps != self.fps:
                print(f"Cam Settings: Simulation FPS is {self.sim_fps} and Camera FPS is {self.fps}")