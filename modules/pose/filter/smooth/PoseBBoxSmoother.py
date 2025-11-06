# Standard library imports
from dataclasses import replace

# Pose imports
from modules.pose.filter.smooth.PoseSmootherBase import PoseSmootherBase
from modules.pose.Pose import Pose, Rect

# Local application imports
from modules.utils.Smoothing import OneEuroFilter


class PoseBBoxSmoother(PoseSmootherBase):
    """Smooths pose bounding boxes using OneEuroFilter."""

    def _create_state(self) -> tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter, OneEuroFilter]:
        """Create filters for bounding box (x, y, width, height)."""
        return (
            OneEuroFilter(self._config.frequency, self._config.min_cutoff, self._config.beta, self._config.d_cutoff),
            OneEuroFilter(self._config.frequency, self._config.min_cutoff, self._config.beta, self._config.d_cutoff),
            OneEuroFilter(self._config.frequency, self._config.min_cutoff, self._config.beta, self._config.d_cutoff),
            OneEuroFilter(self._config.frequency, self._config.min_cutoff, self._config.beta, self._config.d_cutoff)
        )

    def _smooth(self, pose: Pose, state: tuple[OneEuroFilter, OneEuroFilter, OneEuroFilter, OneEuroFilter]) -> Pose:
        """Smooth bounding box for one pose."""
        x_filter, y_filter, w_filter, h_filter = state

        # Smooth bounding box
        smoothed_x = x_filter(pose.bounding_box.x)
        smoothed_y = y_filter(pose.bounding_box.y)
        smoothed_w = w_filter(pose.bounding_box.width)
        smoothed_h = h_filter(pose.bounding_box.height)

        smoothed_bbox = Rect(smoothed_x, smoothed_y, smoothed_w, smoothed_h)
        return replace(pose, bounding_box=smoothed_bbox)

    def _on_config_changed(self) -> None:
        """Update filter parameters when config changes."""
        if self._state is not None:
            x_filter, y_filter, w_filter, h_filter = self._state
            for f in (x_filter, y_filter, w_filter, h_filter):
                f.setParameters(
                    self._config.frequency,
                    self._config.min_cutoff,
                    self._config.beta,
                    self._config.d_cutoff
                )