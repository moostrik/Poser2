from modules.settings import BaseSettings, Field


class AudioSettings(BaseSettings):
    music_id: Field[str] = Field("none", access=Field.READ, description="Music track sent to the Max patch")
