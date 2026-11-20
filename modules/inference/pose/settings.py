from ..engine import ModelWorkerSettings
from modules.settings import Field

# The one pose model, RTMPose-l at 288x384, run in FP16 with its head's norms in FP32 (made from the FP32 original,
# `rtmpose-l_384x288.onnx`, by `tools/models/convert_onnx_to_fp16.py`)
_MODEL: str = "rtmpose-l_384x288_fp16"
_WIDTH: int = 288
_HEIGHT: int = 384


class PoseEstimatorSettings(ModelWorkerSettings):
    """The pose model's settings (RTMPose)."""

    enabled:   Field[bool]       = Field(True)
    max_players:  Field[int]        = Field(1, min=1, max=16, access=Field.INIT)
    dropped:    Field[int]        = Field(0, access=Field.READ, description="Batches replaced by a newer one before the model ran them")

    @property
    def width(self) -> int:
        return _WIDTH

    @property
    def height(self) -> int:
        return _HEIGHT

    @property
    def model(self) -> str:
        """The model's name; its file is `model_file(model)`."""
        return _MODEL
