"""The face detector's settings: the runtime and the model files the other models have, the detection threshold, and
the pitch zero the head pose is measured from."""

from modules.settings import Field

from ..engine import ModelType, ModelWorkerSettings

DETECTION_MODEL: str = 'scrfd_10g_bnkps'
LANDMARKS_MODEL: str = '2d106det'


class FaceDetectorSettings(ModelWorkerSettings):
    model_type:  Field[ModelType] = Field(ModelType.ONNX, access=Field.INIT, description="Runtime the models run on")
    threshold:   Field[float] = Field(0.5, min=0.1, max=1.0, step=0.05, description="Minimum detector score of a face")
    pitch_zero:  Field[float] = Field(0.48, min=0.3, max=0.7, step=0.01, description="Nose position between eyes (0) and mouth (1) that reads as pitch 0", access=Field.READWRITE)
    dropped:     Field[int]   = Field(0, access=Field.READ, description="Images replaced by a newer one before the model ran them")
