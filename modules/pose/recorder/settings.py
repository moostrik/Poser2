from modules.settings import BaseSettings, Field, Widget
from modules.pose.features import Feature


class RecorderSettings(BaseSettings):

    # ── Paths ─────────────────────────────────────────────────────────
    output_path: Field[str] = Field("recordings", access=Field.INIT, description="Pose recordings directory")

    # ── Controls ──────────────────────────────────────────────────────
    record:       Field[bool]  = Field(False, widget=Widget.toggle, description="Record")
    split:        Field[bool]  = Field(False, widget=Widget.button, description="Split chunk")
    name:         Field[str]   = Field("", widget=Widget.input, description="Recording name")
    chunk_length: Field[float] = Field(0.0, min=0.0, max=300.0, description="Chunk duration (seconds)")

    # ── Feature selection ─────────────────────────────────────────────
    features: Field[list] = Field([Feature["BBox"], Feature["Points2D"]], description="Features to include in recording", newline=True)
