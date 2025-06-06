

from enum import Enum

CAMERA_FOV:float = 120.0
TARGET_FOV:float = 90.0

ANGLE_RANGE:float       = 10
VERTICAL_RANGE:float    = 0.05
SIZE_RANGE:float        = 0.05

class FilterType(Enum):
    NONE = 0
    EDGE = 1
    OVERLAP = 2
    DOUBLE = 3