from enum import IntEnum, auto


class ModelType(IntEnum):
    NONE =  0
    ONNX =  auto()
    TRT =   auto()


class Resolution(IntEnum):
    """Model resolution presets."""
    STANDARD = 0
    HIGH = auto()
    ULTRA = auto()
    EXTREME = auto()


RESOLUTION_DIMS = {
    Resolution.STANDARD: (192, 256),  # (width, height)
    Resolution.HIGH: (288, 384),
    Resolution.ULTRA: (384, 512),
    Resolution.EXTREME: (768, 1024),
}

ASPECT_RATIOS = {res: w / h for res, (w, h) in RESOLUTION_DIMS.items()}
