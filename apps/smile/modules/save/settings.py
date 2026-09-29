from modules.settings import BaseSettings, Field, Widget


class ImgFolderSettings(BaseSettings):
    active:     Field[bool] = Field(True, widget=Widget.switch, description="Save captured sequences to images/")
    min_files:  Field[int]  = Field(16, min=1, max=20, step=1, description="Minimum images for a sequence to be kept")
