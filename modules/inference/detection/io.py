from dataclasses import dataclass, field
from typing import Callable

from modules.utils import Rect


@dataclass(frozen=True)
class Detection:
    """The people found in one frame: boxes normalised to the frame, with their scores."""
    cam_id: int
    boxes: list[Rect] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


DetectionCallback = Callable[[Detection], None]
