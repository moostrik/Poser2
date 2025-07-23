# Standard library imports
import math

# Local application imports
from modules.utils.PointsAndRects import Rect, Point2f
from modules.tracker.Tracklet import Tracklet

from modules.utils.HotReloadMethods import HotReloadMethods

class OnePerCamSmoothRect:
    def __init__(self, num_cams: int) -> None:
        self.num_cams: int = num_cams
        self.smooth_rects: list[Rect] = [Rect(0, 0, 0, 0) for _ in range(num_cams)]

        hot_reload = HotReloadMethods(self.__class__)

    def update(self, tracklet: Tracklet) -> Rect:
        smooth_rect = Rect(
            tracklet.roi.x, tracklet.roi.y,
            tracklet.roi.width, tracklet.roi.height
        )
        return smooth_rect
