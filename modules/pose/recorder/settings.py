from modules.settings import BaseSettings, Field, Widget
from modules.pose.features import Feature


class RecorderSettings(BaseSettings):

    # ── Paths ─────────────────────────────────────────────────────────
    output_path: Field[str] = Field("recordings", description="Pose recordings directory")

    # ── Controls ──────────────────────────────────────────────────────
    record:       Field[bool]  = Field(False, widget=Widget.toggle, description="Record")
    split:        Field[bool]  = Field(False, widget=Widget.button, description="Split chunk")
    name:         Field[str]   = Field("", widget=Widget.input, description="Recording name")

    # ── Feature selection ─────────────────────────────────────────────
    features: Field[list] = Field([Feature["BBox"], Feature["Points2D"]], description="Features to include in recording", newline=True)
