from modules.settings import BaseSettings, Field, Group, Widget
from modules.osc.settings import OscSettings


class AudioSettings(BaseSettings):
    # the Max panel's controls: the panel is their source, SMILE sends them to Max every frame
    sound:           Field[bool]  = Field(True, widget=Widget.switch, description="Audio on in the Max patch")
    music_volume:    Field[float] = Field(0.0, min=-40.0, max=6.0, step=0.5, description="Music level in the Max patch, dB")
    laughter:        Field[bool]  = Field(True, widget=Widget.switch, newline=True, description="Laughter samples in the Max patch")
    laughter_volume: Field[float] = Field(0.0, min=-40.0, max=6.0, step=0.5, description="Laughter level in the Max patch, dB")
    music_id:        Field[str]   = Field("none", access=Field.READ, newline=True, description="Music track sent to the Max patch")
    osc:             Group[OscSettings] = Group(OscSettings)
