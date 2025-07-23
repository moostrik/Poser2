# Standard library imports
import math
from dataclasses import dataclass, field

# Local application imports
from modules.utils.PointsAndRects import Rect, Point2f
from modules.tracker.Tracklet import Tracklet

from modules.utils.HotReloadMethods import HotReloadMethods


class OnePerCamSmoothRect:
    def __init__(self, num_cams: int, input_aspect_ratio: float = 16/9, output_aspect_ratio: float = 9/16) -> None:
        self.num_cams: int = num_cams
        self.input_aspect_ratio: float = input_aspect_ratio
        self.output_aspect_ratio: float = output_aspect_ratio

        self.smooth_factor: float = 0.1
        self._limit_center: Point2f = Point2f(0.05, 0.05)  # Maximum normalized change per frame (5% of screen)
        self.limit_height: float = 0.03  # Maximum normalized height change per frame (3% of screen)
        self.smooth_rects: dict[int, Rect] = {}

        hot_reload = HotReloadMethods(self.__class__)

    @property
    def limit_center(self) -> float:
        return self._limit_center.y

    @limit_center.setter
    def limit_center(self, value: float) -> None:
        self._limit_center.x = value
        self._limit_center.y = value / self.input_aspect_ratio


    def update(self, tracklet: Tracklet) -> Rect:
        if tracklet.is_removed:
            if tracklet.id in self.smooth_rects:
                del self.smooth_rects[tracklet.id]

        if tracklet.is_active:
            # Current measurements
            current_center: Point2f = tracklet.roi.center
            current_height: float = tracklet.roi.height

            if tracklet.id in self.smooth_rects:
                prev_rect: Rect = self.smooth_rects[tracklet.id]

                # Limit center changes (normalized 0-1)
                limited_center_x: float = self._clamp_change(current_center.x, prev_rect.center.x, self._limit_center.x)
                limited_center_y: float = self._clamp_change(current_center.y, prev_rect.center.y, self._limit_center.y)

                # Limit height changes (normalized 0-1)
                limited_height: float = self._clamp_change(current_height, prev_rect.height, self.limit_height)

                # Apply smoothing to limited values
                smooth_center_x: float = prev_rect.center.x * (1 - self.smooth_factor) + limited_center_x * self.smooth_factor
                smooth_center_y: float = prev_rect.center.y * (1 - self.smooth_factor) + limited_center_y * self.smooth_factor
                smooth_height: float = prev_rect.height * (1 - self.smooth_factor) + limited_height * self.smooth_factor

                smooth_width: float = smooth_height * self.output_aspect_ratio

                smooth_rect = Rect(
                    smooth_center_x, smooth_center_y,
                    smooth_width, smooth_height
                )
            else:
                # First frame - no smoothing
                smooth_rect = Rect(
                    current_center.x, current_center.y,
                    current_height * self.output_aspect_ratio, current_height
                )

            self.smooth_rects[tracklet.id] = smooth_rect
            return smooth_rect

        return Rect(0, 0, 0, 0)

    def _clamp_change(self, current: float, previous: float, max_change: float) -> float:
        """Limit the change between current and previous normalized values"""
        change = current - previous
        if abs(change) > max_change:
            change = max_change if change > 0 else -max_change
        return previous + change