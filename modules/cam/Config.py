from dataclasses import dataclass, field
from enum import Enum

from modules.cam.depthcam.Definitions import FrameType
from modules.ConfigBase import ConfigBase

@dataclass
class Config(ConfigBase):
    class CoderType(Enum):
        CPU =   0
        GPU =   1
        iGPU =  2

    class CoderFormat(Enum):
        H264 = '.mp4'
        H265 = '.hevc'

    # CAMERA SETTINGS
    ids: list[str]          = field(default_factory=list, metadata={"description": "Camera device IDs"})
    fps: float              = field(default=30.0, metadata={"min": 1.0, "max": 120.0, "description": "Camera frame rate"})
    yolo: bool              = field(default=True, metadata={"description": "Enable YOLO person detection"})
    color: bool             = field(default=True, metadata={"description": "Enable color capture"})
    square: bool            = field(default=True, metadata={"description": "Use square aspect ratio"})
    stereo: bool            = field(default=False, metadata={"description": "Enable stereo mode"})
    hd_ready: bool          = field(default=False, metadata={"description": "Use HD resolution"})
    show_stereo: bool       = field(default=False, metadata={"description": "Show stereo visualization"})
    manual: bool            = field(default=False, metadata={"description": "Manual camera control"})
    flip_h: bool            = field(default=False, metadata={"description": "Flip horizontal"})
    flip_v: bool            = field(default=False, metadata={"description": "Flip vertical"})
    perspective: float      = field(default=0.0, metadata={"min": -1.0, "max": 1.0, "description": "Perspective correction"})

    model_path: str         = field(default="models", metadata={"description": "Model files directory"})
    video_path: str         = field(default="recordings", metadata={"description": "Video recordings directory"})
    temp_path: str          = field(default="temp", metadata={"description": "Temporary files directory"})

    sim_enabled: bool       = field(default=False, metadata={"description": "Enable simulation mode"})
    sim_passthrough: bool   = field(default=False, metadata={"description": "Simulation passthrough"})
    sim_fps: float          = field(default=30.0, metadata={"min": 1.0, "max": 120.0, "description": "Simulation frame rate"})

    rec_enabled: bool       = field(default=False, metadata={"description": "Enable video recording"})

    video_chunk_length: float = field(default=10.0, metadata={"min": 1.0, "max": 300.0, "description": "Video chunk duration (seconds)"})
    video_encoder: CoderType  = field(default=CoderType.iGPU, metadata={"description": "Video encoder type"})
    video_decoder: CoderType  = field(default=CoderType.iGPU, metadata={"description": "Video decoder type"})
    video_format: CoderFormat = field(default=CoderFormat.H264, metadata={"description": "Video format"})
    video_frame_types: list[FrameType] = field(default_factory=lambda: [FrameType.VIDEO], metadata={"description": "Frame types to record"})

    @property
    def num(self) -> int:
        return len(self.ids)

    def __post_init__(self) -> None:
        if self.sim_enabled:
            if self.rec_enabled:
                print("Cam Settings: Warning - Both simulation and recording are enabled, disabling recording.")
                self.rec_enabled = False
            if self.sim_fps != self.fps:
                print(f"Cam Settings: Simulation FPS is {self.sim_fps} and Camera FPS is {self.fps}")