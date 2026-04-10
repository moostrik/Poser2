from modules.settings import BaseSettings, Field, Widget


class SessionSettings(BaseSettings):

    output_path:  Field[str]   = Field("recordings", description="Recordings output directory", access=Field.INIT)
    group_id:     Field[str]   = Field("", widget=Widget.input, description="Recording group ID")
    record:       Field[bool]  = Field(False, widget=Widget.toggle, description="Record")
    split:        Field[bool]  = Field(False, widget=Widget.button, description="Split chunk", visible=False)
    split_seconds: Field[float] = Field(10, min=1, max=60, widget=Widget.number, description="Split recording into chunks of this length (seconds)")