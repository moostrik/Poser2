"""The face detector's settings: the runtime and the model files the other models have, the detection threshold, and
the pitch zero the head pose is measured from."""

from modules.settings import BaseSettings, Field

from ..model_types import ModelType

DETECTION_MODEL: str = 'scrfd_10g_bnkps'
LANDMARKS_MODEL: str = '2d106det'


class FaceDetectorSettings(BaseSettings):
    model_type:  Field[ModelType] = Field(ModelType.ONNX, access=Field.INIT, description="Runtime the models run on")
    model_path:  Field[str]   = Field("data/models", access=Field.INIT, visible=False, description="ONNX model files directory")
    engine_path: Field[str]   = Field("data/models", access=Field.INIT, visible=False, description="TensorRT engines directory")
    threshold:   Field[float] = Field(0.5, min=0.1, max=1.0, step=0.05, description="Minimum detector score of a face")
    pitch_zero:  Field[float] = Field(0.48, min=0.3, max=0.7, step=0.01, description="Nose position between eyes (0) and mouth (1) that reads as pitch 0")
    verbose:     Field[bool]  = Field(False, access=Field.INIT)

    def model_file(self, name: str) -> str:
        """The file of the model `name` for the runtime in use."""
        if self.model_type is ModelType.ONNX:
            return f"{self.model_path}/{name}.onnx"
        return f"{self.engine_path}/{name}.trt"
