from modules.settings import BaseSettings, Field, Group, Widget
from modules.inout import OscSenderSettings


class AudioSettings(BaseSettings):
    # the Max panel's controls: the panel is their source, SMILE sends them to Max every frame
    sound:           Field[bool]  = Field(True, widget=Widget.switch, description="Audio on in the Max patch")
    laughter:        Field[bool]  = Field(True, widget=Widget.switch, description="Laughter samples in the Max patch")
    music_volume:    Field[float] = Field(0.0, min=-40.0, max=6.0, step=0.5, widget=Widget.knob, description="Music level in the Max patch, dB")
    laughter_volume: Field[float] = Field(0.0, min=-40.0, max=6.0, step=0.5, widget=Widget.knob, description="Laughter level in the Max patch, dB")
    laughter_offset: Field[float] = Field(0.0, min=-180.0, max=180.0, step=1.0, widget=Widget.knob, description="Turn of the laughter positions around the sphere, degrees")
    # the laughter test: LaughterTest steps a laugh round the screens, lighting each on the walls
    laughter_test:   Field[bool]  = Field(False, widget=Widget.switch, description="Step one laugh round the screens, lighting each with its number; the clips' laughs are off meanwhile")
    laughter_test_step: Field[float] = Field(2.0, min=0.5, max=10.0, step=0.5, description="Seconds per screen in the laughter test")
    laughter_samples: Field[list[str]] = Field([""], access=Field.READ, visible=False, description="Files in data/audio/laughter, in the patch's order")
    laughter_test_sample: Field[str] = Field("", widget=Widget.text_select, options=laughter_samples, description="Laughter the test plays, a file of data/audio/laughter")
    laughter_test_screen: Field[int] = Field(0, access=Field.READ, description="Screen under test (1-48, table index + 1), 0 while off")
    music_id:        Field[str]   = Field("none", access=Field.READ, description="Music track sent to the Max patch")
    osc:             Group[OscSenderSettings] = Group(OscSenderSettings)
