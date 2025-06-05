
from threading import Lock
from modules.cam.depthcam.Definitions import Tracklet, Rect, Point3f, FrameType


class IdPool:
    def __init__(self, max_size: int) -> None:
        self._available = set(range(max_size))
        self._lock = Lock()

    def acquire(self) -> int:
        with self._lock:
            if not self._available:
                raise Exception("No more IDs available")
            return self._available.pop()

    def release(self, obj: int) -> None:
        with self._lock:
            self._available.add(obj)

    def size(self) -> int:
        return len(self._available)




