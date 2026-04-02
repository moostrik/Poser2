from modules.settings import Settings, Field


class SyncSettings(Settings):
    num_cameras:    Field[int]   = Field(1, access=Field.INIT)
    fps:            Field[float] = Field(30.0, access=Field.INIT)
