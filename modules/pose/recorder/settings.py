from modules.settings import BaseSettings, Field, Widget
from modules.pose.features import Feature


class RecorderSettings(BaseSettings):

    # ── Controls ──────────────────────────────────────────────────────
    enabled:      Field[bool]  = Field(True, widget=Widget.switch, description="Enable pose recording")
    start:        Field[bool]  = Field(False, widget=Widget.button, description="Start recording")
    stop:         Field[bool]  = Field(False, widget=Widget.button, description="Stop recording")
    recording:    Field[bool]  = Field(False, access=Field.READ, description="Recording")
    split:        Field[bool]  = Field(False, widget=Widget.button, description="Split chunk")
    output_path:  Field[str]   = Field("recordings", access=Field.INIT, description="Pose recordings directory")
    name:         Field[str]   = Field("", widget=Widget.input, description="Recording name")

    # ── Recording options ─────────────────────────────────────────────
    stage:    Field[int] = Field(0, description="Pipeline stage to record")
    features: Field[list]  = Field([Feature["BBox"], Feature["Points2D"]], description="Features to include in recording", newline=True)
