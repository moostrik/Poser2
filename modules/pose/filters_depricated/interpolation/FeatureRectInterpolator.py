# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.filters_depricated.interpolation.FeatureInterpolatorBase import FeatureInterpolatorBase

# Local application imports
from modules.utils.Interpolation import VectorPredictiveHermite
from modules.utils.PointsAndRects import Rect


@dataclass
class RectFilterState:
    """State for rect interpolation."""
    interpolator: VectorPredictiveHermite


class FeatureRectInterpolator(FeatureInterpolatorBase[Rect]):
    """Interpolator for bounding box rectangles.

    Features:
    - Smooth interpolation of x, y, width, height
    - Velocity-based smoothing for natural motion using Hermite interpolation
    - Ensures non-negative width/height after interpolation

    Note:
        Uses VectorPredictiveHermite internally for 4D vector [x, y, w, h].
    """

    def _create_state(self) -> RectFilterState:
        """Create initial filter state for interpolation."""
        return RectFilterState(
            interpolator=VectorPredictiveHermite(
                input_rate=self._input_rate,
                vector_size=4,  # [x, y, width, height]
                alpha_v=self._alpha_v
            )
        )

    def _add_sample(self, feature: Rect) -> None:
        """Add rect sample to interpolator."""
        state: RectFilterState = self._state
        values = np.array([feature.x, feature.y, feature.width, feature.height], dtype=np.float32)
        state.interpolator.add_sample(values)

    def _interpolate(self, current_time: float | None) -> Rect:
        """Generate interpolated rect at current time."""
        state: RectFilterState = self._state

        # Update interpolator to current time
        state.interpolator.update(current_time)

        # Get interpolated values
        values: np.ndarray = state.interpolator.interpolated_value

        return Rect(
            x=float(values[0]),
            y=float(values[1]),
            width=max(0.0, float(values[2])),   # Ensure non-negative
            height=max(0.0, float(values[3]))   # Ensure non-negative
        )

    def _on_alpha_v_changed(self, value: float) -> None:
        """Update interpolator when alpha_v changes."""
        self._state.interpolator.alpha_v = value