from dataclasses import replace

from modules.pose.Pose import Pose, Rect
from modules.pose.filters.smooth.PoseSmoothFilterBase import PoseSmoothFilterBase
from modules.utils.Smoothing import OneEuroFilter
from modules.Settings import Settings


class PoseSmoothBBoxFilter(PoseSmoothFilterBase):
    """Smooths pose bounding boxes using OneEuroFilter."""

    def _create_tracklet_state(self) -> tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter, OneEuroFilter]:
        """Create filters for bounding box (x, y, width, height)."""
        return (
            OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
            OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
            OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
            OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
        )

    def _smooth_pose(self, pose: Pose, tracklet_id: int) -> Pose:
        """Smooth bounding box for one pose."""
        x_filter, y_filter, w_filter, h_filter = self._tracklets[tracklet_id]
        timestamp: float = pose.time_stamp.timestamp()

        # Smooth bounding box
        smoothed_x = x_filter(pose.bounding_box.x, timestamp)
        smoothed_y = y_filter(pose.bounding_box.y, timestamp)
        smoothed_w = w_filter(pose.bounding_box.width, timestamp)
        smoothed_h = h_filter(pose.bounding_box.height, timestamp)

        smoothed_bbox = Rect(smoothed_x, smoothed_y, smoothed_w, smoothed_h)
        return replace(pose, bounding_box=smoothed_bbox)

    def _update_tracklet_filters(self, tracklet_state: tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter, OneEuroFilter]) -> None:
        """Update filter parameters for bounding box filters."""
        x_filter, y_filter, w_filter, h_filter = tracklet_state
        x_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
        y_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
        w_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
        h_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)