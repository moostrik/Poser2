from modules.settings import BaseSettings, Field, Widget


class SessionSettings(BaseSettings):

    record:       Field[bool]  = Field(False, widget=Widget.toggle, description="Record")
    split:        Field[bool]  = Field(False, widget=Widget.button, description="Split chunk")
    group_id:     Field[str]   = Field("", widget=Widget.input, description="Recording group ID")
    chunk_length: Field[float] = Field(10.0, min=1.0, max=300.0, description="Chunk duration (seconds)")
