"""The person detector's settings: the model file, and the two thresholds the device detector has."""

from modules.settings import BaseSettings, Field


class Settings(BaseSettings):
    """A YOLO person detector on the host (ONNX Runtime), the replay backend's stand-in for the
    device's detector; `confidence` and `overlap` default to the values the oak pipeline sets."""

    model:      Field[str]   = Field("yolov8n_coco_512x448.onnx", access=Field.INIT, description="ONNX model file in model_path")
    model_path: Field[str]   = Field("data/models", access=Field.INIT, visible=False, description="Model files directory")
    confidence: Field[float] = Field(0.5, min=0.0, max=1.0, step=0.05, description="Minimum person score")
    overlap:    Field[float] = Field(0.5, min=0.0, max=1.0, step=0.05, description="Boxes overlapping more than this (IoU) are one person")
    verbose:    Field[bool]  = Field(False, access=Field.INIT)

    @property
    def model_file(self) -> str:
        return f"{self.model_path}/{self.model}"
