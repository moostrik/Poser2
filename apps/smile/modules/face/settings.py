from modules.settings import BaseSettings, Field


class FaceSettings(BaseSettings):
    size_filter:   Field[float] = Field(0.1, min=0.0, max=0.3, step=0.01, description="Minimum face height, relative to the image height")
    confidence:    Field[float] = Field(0.7, min=0.5, max=1.0, step=0.01, description="Minimum detector score")
    stable_frames: Field[int]   = Field(3, min=1, max=10, step=1, description="Frames in a row a face must be found to start a sequence")
    zoom:          Field[float] = Field(1.2, min=1.0, max=5.0, step=0.1, description="Portrait width, relative to the face height")
    offset:        Field[float] = Field(0.1, min=-0.3, max=0.3, step=0.01, description="Portrait vertical offset, relative to the face height")
    result:        Field[str]   = Field("", access=Field.READ, description="Last detection result")
