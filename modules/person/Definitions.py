CAM_360_FOV:float =                 120.0
CAM_360_TARGET_FOV:float =          90.0
CAM_360_EDGE_THRESHOLD: float =     0.5     # threshold for radial edge filter
CAM_360_OVERLAP_EXPANSION: float =  0.3     # expansion factor for radial overlap filter
CAM_360_HYSTERESIS_FACTOR: float =  0.9     # hysteresis factor for radial overlap filter

PERSON_ROI_EXPANSION: float =       0.1
PERSON_TIMEOUT: float =             1.0     # seconds

MIN_TRACKLET_AGE: int =             4       # frames, minimum age of a person to be considered active
MIN_TRACKLET_HEIGHT: float =        0.25    # normalized height, minimum size for pose detection


from threading import Lock

class IdPool:
    def __init__(self, max_size: int) -> None:
        self._available = set(range(max_size))
        self._lock = Lock()

    def acquire(self) -> int:
        with self._lock:
            if not self._available:
                raise Exception("No more IDs available")
            min_id: int = min(self._available)
            self._available.remove(min_id)
            return min_id

    def release(self, obj: int) -> None:
        with self._lock:
            self._available.add(obj)

    def size(self) -> int:
        return len(self._available)