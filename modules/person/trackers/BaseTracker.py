

from enum import Enum
from typing import Protocol

class BaseTrackerInfo(Protocol):
    pass


class TrackingStatus(Enum):
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3
    NONE = 4