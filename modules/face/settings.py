from modules.settings import BaseSettings, Field, Widget


class FaceSettings(BaseSettings):
    confidence:    Field[float] = Field(0.7, min=0.5, max=1.0, step=0.01, newline=True, description="Minimum detector score")

    eye_min:       Field[float] = Field(0.05, min=0.0, max=0.3, step=0.01, newline=True, label="Min eye dist", description="Minimum distance between the eyes (fraction of the image width)")
    crop_margin:   Field[float] = Field(0.0, min=-0.5, max=0.5, step=0.01, description="Minimum crop distance to the image edge (crop widths); negative reaches past")
    eye_open:      Field[float] = Field(0.2, min=0.0, max=0.5, step=0.01, label="Min eye open", description="Minimum eye opening, height relative to the eye width")

    max_yaw:       Field[float] = Field(20.0, min=0.0, max=60.0, step=1.0, newline=True, description="Maximum head turn left or right (degrees, approximate)")
    max_pitch:     Field[float] = Field(25.0, min=0.0, max=60.0, step=1.0, description="Maximum head tilt up or down (degrees, approximate)")
    max_roll:      Field[float] = Field(30.0, min=0.0, max=90.0, step=1.0, description="Maximum head tilt sideways (degrees)")

    falloff:       Field[float] = Field(1.0, min=0.1, max=3.0, step=0.1, newline=True, label="Far at", description="Distance past a limit that counts as fully far (fraction of the limit)")

    pitch_zero:   Field[float] = Field(0.48, min=0.3, max=0.7, step=0.01, newline=True, description="Nose position between eyes (0) and mouth (1) that reads as pitch 0")
    calibrate:     Field[bool]  = Field(False, widget=Widget.button, label="Set pitch zero", description="Take the pitch zero from the current face: look straight at the camera")

    zoom:          Field[float] = Field(1.2, min=1.0, max=5.0, step=0.01, newline=True, label="Portrait zoom", description="Portrait width, relative to the face height")
    offset:        Field[float] = Field(0.0, min=-0.3, max=0.3, step=0.01, label="Portrait offset", description="Portrait vertical offset, relative to the face height")

    result:        Field[str]   = Field("", access=Field.READ, widget=Widget.textarea, newline=True, description="Last detection result")
