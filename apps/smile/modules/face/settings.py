from modules.settings import BaseSettings, Field


class FaceSettings(BaseSettings):
    size_filter:   Field[float] = Field(0.1, min=0.0, max=0.3, step=0.01, description="Minimum face height, relative to the image height")
    confidence:    Field[float] = Field(0.7, min=0.5, max=1.0, step=0.01, description="Minimum detector score")
    max_yaw:       Field[float] = Field(20.0, min=0.0, max=60.0, step=1.0, description="Maximum head turn left or right (degrees, approximate)")
    max_pitch:     Field[float] = Field(25.0, min=0.0, max=60.0, step=1.0, description="Maximum head tilt up or down (degrees, approximate)")
    max_roll:      Field[float] = Field(30.0, min=0.0, max=90.0, step=1.0, description="Maximum head tilt sideways (degrees)")
    eye_open:      Field[float] = Field(0.2, min=0.0, max=0.5, step=0.01, description="Minimum eye opening, height relative to the eye width")
    stable_frames: Field[int]   = Field(3, min=1, max=10, step=1, description="Frames in a row a face must qualify to start a sequence")
    window:        Field[float] = Field(1.0, min=0.1, max=5.0, step=0.1, description="Time before the flash in which the best face is picked (s)")
    zoom:          Field[float] = Field(1.2, min=1.0, max=5.0, step=0.01, description="Portrait width, relative to the face height")
    offset:        Field[float] = Field(0.0, min=-0.3, max=0.3, step=0.01, description="Portrait vertical offset, relative to the face height")
    result:        Field[str]   = Field("", access=Field.READ, description="Last detection result")
