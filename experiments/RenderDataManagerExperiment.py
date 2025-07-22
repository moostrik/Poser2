import numpy as np
from itertools import combinations
from typing import Optional, Tuple, Dict, List
from threading import Lock
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic, Any


from modules.cam.depthcam.Definitions import Tracklet as DepthTracklet
from modules.tracker.Tracklet import Tracklet
from modules.pose.PoseDefinitions import Pose
from modules.pose.PoseStream import PoseStreamData
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.av.Definitions import AvOutput

DATA_SCHEMA = [
    ("av_frame", "single", Optional[AvOutput]),
    ("cam_frames", "dict", np.ndarray),
    ("depth_tracklets", "dict", List[DepthTracklet]),
    ("tracklets", "dict", Tracklet),
    ("poses", "dict", Pose),
    ("pose_streams", "dict", PoseStreamData),
    ("r_streams", "single", Optional[PairCorrelationStreamData]),
]

T = TypeVar("T")
@dataclass
class DataItem(Generic[T]):
    value: T
    dirty: bool

class RenderDataManager:
    def __init__(self) -> None:
        self.mutex = Lock()
        for name, kind, _ in DATA_SCHEMA:
            if kind == "single":
                setattr(self, name, DataItem(None, False))
            elif kind == "dict":
                setattr(self, name, {})

def make_setter(name, kind):
    def setter(self, key_or_value, value=None):
        with self.mutex:
            if kind == "single":
                getattr(self, name).value = key_or_value
                getattr(self, name).dirty = True
            elif kind == "dict":
                if key_or_value not in getattr(self, name):
                    getattr(self, name)[key_or_value] = DataItem(None, False)
                getattr(self, name)[key_or_value].value = value
                getattr(self, name)[key_or_value].dirty = True
    return setter

def make_getter(name, kind):
    def getter(self, key=None, only_if_dirty=True, mark_clean=True) -> Optional[Any]:
        with self.mutex:
            if kind == "single":
                item: DataItem[Any] = getattr(self, name)
                if only_if_dirty and not item.dirty:
                    return None
                if mark_clean:
                    item.dirty = False
                return item.value
            elif kind == "dict":
                item = getattr(self, name).get(key)
                if not item:
                    return None
                if only_if_dirty and not item.dirty:
                    return None
                if mark_clean:
                    item.dirty = False
                return item.value
    return getter

for name, kind, _ in DATA_SCHEMA:
    setattr(RenderDataManager, f"set_{name}", make_setter(name, kind))
    setattr(RenderDataManager, f"get_{name}", make_getter(name, kind))