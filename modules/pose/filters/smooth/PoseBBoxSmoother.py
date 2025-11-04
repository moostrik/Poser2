from dataclasses import replace

from modules.pose.Pose import Pose, Rect
from modules.pose.filters.smooth.PoseSmootherBase import PoseSmootherBase
from modules.utils.Smoothing import OneEuroFilter


class PoseBBoxSmoother(PoseSmootherBase):
    """Smooths pose bounding boxes using OneEuroFilter."""

    def _create_tracklet_state(self) -> tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter, OneEuroFilter]:
        """Create filters for bounding box (x, y, width, height)."""
        return (
            OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
            OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
            OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff),
            OneEuroFilter(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
        )

    def _smooth(self, pose: Pose, tracklet_id: int) -> Pose:
        """Smooth bounding box for one pose."""
        x_filter, y_filter, w_filter, h_filter = self._tracklets[tracklet_id]

        # Smooth bounding box
        smoothed_x = x_filter(pose.bounding_box.x)
        smoothed_y = y_filter(pose.bounding_box.y)
        smoothed_w = w_filter(pose.bounding_box.width)
        smoothed_h = h_filter(pose.bounding_box.height)

        smoothed_bbox = Rect(smoothed_x, smoothed_y, smoothed_w, smoothed_h)
        return replace(pose, bounding_box=smoothed_bbox)

    def _update_tracklet_filters(self, tracklet_state: tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter, OneEuroFilter]) -> None:
        """Update filter parameters for bounding box filters."""
        x_filter, y_filter, w_filter, h_filter = tracklet_state
        x_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
        y_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
        w_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)
        h_filter.setParameters(self.settings.frequency, self.settings.min_cutoff, self.settings.beta, self.settings.d_cutoff)