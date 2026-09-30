from modules.settings import BaseSettings, Field, Widget


class CameraSettings(BaseSettings):
    flip_h:       Field[bool] = Field(False, description="Mirror the camera image horizontally")
    flip_v:       Field[bool] = Field(False, description="Flip the camera image vertically")
    gain_auto:    Field[bool] = Field(False, widget=Widget.switch, description="Vimba: automatic gain")
    gain_target:  Field[int]  = Field(40, min=0, max=100, step=1, description="Vimba: auto gain target brightness")
    image_select: Field[int]  = Field(0, min=0, step=1, widget=Widget.number, description="Image camera: index of the .jpg in input/, sorted by name")
