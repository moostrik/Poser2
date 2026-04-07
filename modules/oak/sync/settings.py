from modules.settings import BaseSettings, Field


class SyncSettings(BaseSettings):
    num_cameras:    Field[int]   = Field(1, access=Field.INIT, visible=False)
    fps:            Field[float] = Field(30.0, access=Field.INIT, visible=False)
