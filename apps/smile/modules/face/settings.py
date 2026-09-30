from modules.settings import BaseSettings, Field, Widget


class FaceSettings(BaseSettings):
    confidence:    Field[float] = Field(0.7, min=0.5, max=1.0, step=0.01, newline=True, row_label="Detect", label="Confidence", description="Minimum detector score")
    size_filter:   Field[float] = Field(0.1, min=0.0, max=0.3, step=0.01, label="Min size", description="Minimum face height, relative to the image height")
    stable_frames: Field[int]   = Field(3, min=1, max=10, step=1, label="Stable frames", description="Frames in a row a face must qualify to start a sequence")

    max_yaw:       Field[float] = Field(20.0, min=0.0, max=60.0, step=1.0, newline=True, row_label="Pose", label="Yaw", description="Maximum head turn left or right (degrees, approximate)")
    max_pitch:     Field[float] = Field(25.0, min=0.0, max=60.0, step=1.0, label="Pitch", description="Maximum head tilt up or down (degrees, approximate)")
    max_roll:      Field[float] = Field(30.0, min=0.0, max=90.0, step=1.0, label="Roll", description="Maximum head tilt sideways (degrees)")
    pitch_zero:    Field[float] = Field(0.48, min=0.3, max=0.7, step=0.01, newline=True, row_label="Pose", label="Pitch zero", description="Nose position between the eyes (0) and the mouth (1) that reads as pitch 0; depends on the camera height")
    calibrate:     Field[bool]  = Field(False, widget=Widget.button, label="Set pitch zero", description="Take the pitch zero from the current face: look straight at the camera")

    eye_open:      Field[float] = Field(0.2, min=0.0, max=0.5, step=0.01, newline=True, row_label="Eyes", label="Min opening", description="Minimum eye opening, height relative to the eye width")

    window:        Field[float] = Field(1.0, min=0.1, max=5.0, step=0.1, newline=True, row_label="Capture", label="Window (s)", description="Time before the flash in which the best face is picked (s)")

    zoom:          Field[float] = Field(1.2, min=1.0, max=5.0, step=0.01, newline=True, row_label="Portrait", label="Zoom", description="Portrait width, relative to the face height")
    offset:        Field[float] = Field(0.0, min=-0.3, max=0.3, step=0.01, label="Offset", description="Portrait vertical offset, relative to the face height")

    result:        Field[str]   = Field("", access=Field.READ, widget=Widget.textarea, newline=True, description="Last detection result")
