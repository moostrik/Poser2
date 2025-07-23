# Standard library imports
import math
from dataclasses import dataclass, field

# Local application imports
from modules.utils.PointsAndRects import Rect, Point2f
from modules.tracker.Tracklet import Tracklet

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class smooth_value:
    center: Point2f
    height: float

class OnePerCamSmoothRect:
    def __init__(self, num_cams: int, input_aspect_ratio: float = 16/9, output_aspect_ratio: float = 9/16) -> None:
        self.num_cams: int = num_cams
        self.input_aspect_ratio: float = input_aspect_ratio
        self.output_aspect_ratio: float = output_aspect_ratio

        self.smooth_factor: float = 0.1
        self._limit_center: Point2f = Point2f(0.05, 0.05)  # Maximum normalized change per frame (5% of screen)
        self.limit_height: float = 0.03  # Maximum normalized height change per frame (3% of screen)
        self.smooth_values: dict[int, smooth_value] = {}
        self.expand_height: float = 0.3  # Expand height by 10% of the screen height

        hot_reload = HotReloadMethods(self.__class__)

    @property
    def limit_center(self) -> float:
        return self._limit_center.y

    @limit_center.setter
    def limit_center(self, value: float) -> None:
        self._limit_center.x = value
        self._limit_center.y = value / self.input_aspect_ratio


    def update(self, tracklet: Tracklet) -> Rect | None:
        if tracklet.is_removed:
            if tracklet.id in self.smooth_values:
                del self.smooth_values[tracklet.id]

        if tracklet.is_active:
            # Current measurements
            current_center: Point2f = tracklet.roi.center
            current_height: float = tracklet.roi.height * (1 + self.expand_height)

            if tracklet.id in self.smooth_values:
                prev_smooth: smooth_value = self.smooth_values[tracklet.id]

                # Limit center changes (normalized 0-1)
                limited_center_x: float = OnePerCamSmoothRect._clamp_change(current_center.x, prev_smooth.center.x, self._limit_center.x)
                limited_center_y: float = OnePerCamSmoothRect._clamp_change(current_center.y, prev_smooth.center.y, self._limit_center.y)

                # Limit height changes (normalized 0-1)
                limited_height: float = OnePerCamSmoothRect._clamp_change(current_height, prev_smooth.height, self.limit_height)

                # Apply smoothing to limited values
                smooth_center_x: float = prev_smooth.center.x * (1 - self.smooth_factor) + limited_center_x * self.smooth_factor
                smooth_center_y: float = prev_smooth.center.y * (1 - self.smooth_factor) + limited_center_y * self.smooth_factor
                smooth_height: float = prev_smooth.height * (1 - self.smooth_factor) + limited_height * self.smooth_factor

                smooth_width: float = smooth_height * self.output_aspect_ratio

                smooth = smooth_value(Point2f(smooth_center_x, smooth_center_y), smooth_height)
            else:
                # First frame - no smoothing
                smooth = smooth_value(Point2f(current_center.x, current_center.y), current_height)

            self.smooth_values[tracklet.id] = smooth

            return OnePerCamSmoothRect._rect_from_smooth(smooth, self.output_aspect_ratio)

        return None

    @staticmethod
    def _clamp_change(current: float, previous: float, max_change: float) -> float:
        """Limit the change between current and previous normalized values"""
        change = current - previous
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
        return previous + change

    @staticmethod
    def _rect_from_smooth(smooth: smooth_value, aspect_ratio: float) -> Rect:
        """Convert smooth value to Rect"""
        width: float = smooth.height * aspect_ratio # * aspect_ratio
        return Rect(
            x=smooth.center.x - width / 2,
            y=smooth.center.y - smooth.height / 2,
            width=width,
            height=smooth.height
        )