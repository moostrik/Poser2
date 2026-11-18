from dataclasses import dataclass, field
from typing import Callable, TypeAlias

from modules.utils import Rect


@dataclass(frozen=True)
class Detection:
    """The people found in one frame: boxes normalised to the frame, with their scores, and the
    frame's `timestamp` as it was submitted."""
    cam_id: int
    timestamp: float
    boxes: list[Rect] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


DetectionCallback: TypeAlias = Callable[[Detection], None]
