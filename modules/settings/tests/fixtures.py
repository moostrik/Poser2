"""Settings classes more than one test file uses."""

from enum import Enum

from modules.settings import Field, Widget, BaseSettings, Group
from modules.utils import Color


class RenderMode(Enum):
    WIREFRAME = "wireframe"
    SOLID = "solid"
    TEXTURED = "textured"


class CameraSettings(BaseSettings):
    exposure = Field(1000, min=100, max=10000, step=100, description="Exposure time (µs)")
    gain = Field(1.0, min=0.0, max=16.0, step=0.1)
    overlay_color = Field(Color(1, 0, 0))
    resolution = Field(1080, access=Field.INIT)
    fps = Field(0.0, access=Field.READ, visible=False)
    mode = Field(RenderMode.SOLID)


class MinimalSettings(BaseSettings):
    value = Field(0)


class SettingsWithActions(BaseSettings):
    exposure = Field(1000)
    reset = Field(False, widget=Widget.button, description="Reset all values")
    hidden_action = Field(False, widget=Widget.button, visible=False)


class InnerSettings(BaseSettings):
    speed = Field(1.0, min=0.0, max=10.0)
    scale = Field(0.5)


class OuterSettings(BaseSettings):
    fps = Field(60.0, min=1.0, max=240.0)
    inner = Group(InnerSettings)


class CoreSettings(BaseSettings):
    """Standalone settings — no knowledge of any parent."""
    fps      = Field(30.0, access=Field.INIT)
    color    = Field(True, access=Field.INIT)
    exposure = Field(1000)


class PlayerSettings(BaseSettings):
    folder = Field("")
    playing = Field(False, access=Field.READ)


class ParentWithChild(BaseSettings):
    fps    = Field(30.0, access=Field.INIT)
    color  = Field(True, access=Field.INIT)
    player = Group(PlayerSettings)
    core_0 = Group(CoreSettings, share=[fps, color])
    core_1 = Group(CoreSettings, share=[fps, color])
    core_2 = Group(CoreSettings, share=[fps, color])

    @property
    def cores(self) -> list[CoreSettings]:
        return [self.core_0, self.core_1, self.core_2]
