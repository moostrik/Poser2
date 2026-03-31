from .camera.settings import CameraSettings
from .camera.definitions import FrameType, CoderFormat
from .simulator.settings import SimulatorSettings
from .recorder.settings import RecorderSettings
from modules.settings import Settings, Field


class OakSettings(Settings):

    # Camera count — source of truth for cores Child count
    num_cameras:    Field[int]              = Field(1, access=Field.INIT, description="Number of cameras")

    # Core camera settings (shared to cores, recorder, player as needed)
    fps:            Field[float]            = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera frame rate")
    yolo:           Field[bool]             = Field(True, access=Field.INIT, description="Enable YOLO person detection")
    color:          Field[bool]             = Field(True, access=Field.INIT, description="Enable color capture")
    square:         Field[bool]             = Field(True, access=Field.INIT, description="Use square aspect ratio")
    stereo:         Field[bool]             = Field(False, access=Field.INIT, description="Enable stereo mode")
    hd_ready:       Field[bool]             = Field(False, access=Field.INIT, description="Use HD resolution")
    sim_enabled:    Field[bool]             = Field(False, access=Field.INIT, description="Enable simulation mode")

    # Paths
    model_path:     Field[str]              = Field("models", access=Field.INIT, visible=False, description="Model files directory")
    video_path:     Field[str]              = Field("recordings", access=Field.INIT, visible=False, description="Video recordings directory")
    temp_path:      Field[str]              = Field("temp", access=Field.INIT, visible=False, description="Temporary files directory")

    # Video settings (shared to player and recorder)
    video_format:   Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, description="Video format")
    video_frame_types: Field[list[FrameType]] = Field([FrameType.VIDEO], access=Field.INIT, description="Frame types to record")

    # Children
    cores: list[CameraSettings]     = CameraSettings(count=num_cameras, share=[fps, color, square, stereo, yolo, hd_ready, sim_enabled, model_path])  # type: ignore[assignment]
    simulator: SimulatorSettings    = SimulatorSettings(share=[video_path, video_format, video_frame_types, num_cameras, color, square, stereo])
    recorder: RecorderSettings      = RecorderSettings(share=[video_path, temp_path, video_format, video_frame_types, color, square, stereo, num_cameras, fps])
