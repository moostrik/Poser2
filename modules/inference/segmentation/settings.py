from enum import IntEnum, auto

from modules.settings import Field

from ..engine import ModelWorkerSettings


class Resolution(IntEnum):
    """The input sizes RVM was exported at."""
    STANDARD = 0
    HIGH = auto()
    ULTRA = auto()
    EXTREME = auto()


RESOLUTION_DIMS: dict[Resolution, tuple[int, int]] = {  # (width, height)
    Resolution.STANDARD: (192, 256),
    Resolution.HIGH:     (288, 384),
    Resolution.ULTRA:    (384, 512),
    Resolution.EXTREME:  (768, 1024),
}

# The model of each resolution RVM was exported at, in FP16
_MODELS: dict[Resolution, str] = {
    Resolution.STANDARD: "rvm_mobilenetv3_256x192_fp16",
    Resolution.HIGH:     "rvm_mobilenetv3_384x288_fp16",
    Resolution.ULTRA:    "rvm_mobilenetv3_512x384_fp16",
    Resolution.EXTREME:  "rvm_mobilenetv3_1024x768_fp16",
}


class SegmenterSettings(ModelWorkerSettings):
    """The person segmentation's settings (RVM)."""

    max_players:      Field[int]        = Field(1, min=1, max=16, access=Field.INIT)
    resolution:     Field[Resolution] = Field(Resolution.STANDARD, access=Field.INIT)
    enabled:        Field[bool]       = Field(True)
    reset_interval: Field[int]        = Field(60, min=1, access=Field.INIT)
    dropped:        Field[int]        = Field(0, access=Field.READ, description="Batches replaced by a newer one before the model ran them")

    @property
    def width(self) -> int:
        return RESOLUTION_DIMS[self.resolution][0]

    @property
    def height(self) -> int:
        return RESOLUTION_DIMS[self.resolution][1]

    @property
    def model(self) -> str:
        """The model's name at the resolution; its file is `model_file(model)`."""
        return _MODELS[self.resolution]
