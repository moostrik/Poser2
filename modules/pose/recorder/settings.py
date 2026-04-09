from modules.settings import BaseSettings, Field, Widget
from modules.pose.features import Feature


class RecorderSettings(BaseSettings):

    # ── Paths ─────────────────────────────────────────────────────────
    recordings_path: Field[str] = Field("recordings", access=Field.INIT, description="Directory for standalone pose recordings")

    # ── Controls ──────────────────────────────────────────────────────

    start:     Field[bool] = Field(False, widget=Widget.button, description="Start recording")
    stop:      Field[bool] = Field(False, widget=Widget.button, description="Stop recording")
    recording: Field[bool] = Field(False, access=Field.READ, widget=Widget.toggle, description="Recording active")

    # ── Feature selection ─────────────────────────────────────────────
    features: Field[list] = Field([Feature["BBox"], Feature["Points2D"]], description="Features to include in recording", newline=True)
