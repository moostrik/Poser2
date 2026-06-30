from ..camera.definitions import FrameType, CoderType, CoderFormat
from modules.settings import BaseSettings, Field, Widget


class SimulatorSettings(BaseSettings):

    # ── Initial settings ───────────────────────────────────────────────
    num_cameras:        Field[int]              = Field(1, access=Field.INIT, description="Number of cameras")
    fps:                Field[float]            = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Simulation FPS")
    video_path:         Field[str]              = Field("recordings", access=Field.INIT, description="Video recordings directory")
    video_decoder:      Field[CoderType]        = Field(CoderType.iGPU, access=Field.INIT, description="Video decoder type")
    video_format:       Field[CoderFormat]      = Field(CoderFormat.H264, access=Field.INIT, description="Video format")
    video_frame_types:  Field[list[FrameType]]  = Field([FrameType.VIDEO], access=Field.INIT, description="Frame types to record", visible=False)

    # ── Playback controls ─────────────────────────────────────────────
    start:              Field[bool]             = Field(False, widget=Widget.button, description="Start playback", newline=True)
    stop:               Field[bool]             = Field(False, widget=Widget.button, description="Stop playback")
    available_folders:  Field[list[str]]        = Field([""], access=Field.READ, visible=False, description="Available recording folders")
    folder:             Field[str]              = Field("", widget=Widget.text_select, options=available_folders, description="Recording folder")
    refresh:            Field[bool]             = Field(False, widget=Widget.button, description="Refresh recording folders")
    sim_passthrough:    Field[bool]             = Field(False, description="Passthrough", visible=False)
    start_norm:         Field[float]            = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, description="Range start", newline=True)
    end_norm:           Field[float]            = Field(1.0, min=0.0, max=1.0, widget=Widget.slider, description="Range end")
    playback_norm:      Field[float]            = Field(0.0, min=0.0, max=1.0, access=Field.READ, widget=Widget.slider, description="Playback head (normalized)")
    start_time:         Field[str]              = Field("0:00.0", access=Field.READ, description="Start time", newline=True)
    end_time:           Field[str]              = Field("0:00.0", access=Field.READ, description="End time")
    playback_time:      Field[str]              = Field("0:00.0", access=Field.READ, description="Playback head")
    current_chunk:      Field[int]              = Field(0, access=Field.READ, description="Current chunk", newline=True)
    max_chunks:         Field[int]              = Field(0, access=Field.READ, description="Total chunks")
