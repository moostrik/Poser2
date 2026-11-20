"""The person detector's settings: the runtime and input size the other models have, and the two
thresholds the device detector has."""

from enum import IntEnum, auto

from modules.settings import Field

from ..engine import ModelType, ModelWorkerSettings


class Resolution(IntEnum):
    """The model's input sizes: the OAK-D blobs' 512x448, and the three the OAK4 detector comes in."""
    R512X448 = 0
    R480X384 = auto()
    R512X416 = auto()
    R640X512 = auto()


RESOLUTION_DIMS: dict[Resolution, tuple[int, int]] = {  # (width, height)
    Resolution.R512X448: (512, 448),
    Resolution.R480X384: (480, 384),
    Resolution.R512X416: (512, 416),
    Resolution.R640X512: (640, 512),
}


class PersonDetectorSettings(ModelWorkerSettings):
    """A YOLO person detector on the host, the replay backend's stand-in for the device's detector;
    `confidence` and `overlap` default to the values the oak pipeline sets."""

    model_type:  Field[ModelType]  = Field(ModelType.ONNX, access=Field.INIT, description="Runtime the model runs on")
    enabled:     Field[bool]  = Field(True, description="Find the players on the host; off, a recording's own tracklets play")
    resolution:  Field[Resolution] = Field(Resolution.R512X448, access=Field.INIT, description="Model input size, width x height")
    confidence:  Field[float] = Field(0.5, min=0.0, max=1.0, step=0.05, description="Minimum person score")
    overlap:     Field[float] = Field(0.5, min=0.0, max=1.0, step=0.05, description="Boxes overlapping more than this (IoU) are one person")
    dropped:     Field[int]   = Field(0, access=Field.READ, description="Images replaced by a newer one before the model ran them")

    @property
    def width(self) -> int:
        return RESOLUTION_DIMS[self.resolution][0]

    @property
    def height(self) -> int:
        return RESOLUTION_DIMS[self.resolution][1]

    @property
    def model(self) -> str:
        """The model's name at the resolution; its file is `model_file(model)`."""
        return f"yolov8n_coco_{self.width}x{self.height}"
