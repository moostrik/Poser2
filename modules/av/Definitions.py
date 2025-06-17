from enum import Enum, IntEnum, auto

class CompMode(IntEnum):
    NONE = auto()
    TEST_1 = auto()
    CALIBRATE_1 = auto()
    COMP_1 = auto()

ANGLE =         3
RESOLUTION =    4000
RATE =          30