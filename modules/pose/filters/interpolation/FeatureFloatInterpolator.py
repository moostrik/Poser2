# Standard library imports
from dataclasses import dataclass

# Pose imports
from .FeatureInterpolatorBase import FeatureInterpolatorBase

# Local application imports
from modules.utils.Interpolation import ScalarPredictiveHermite


@dataclass
class FloatFilterState:
    """State for float interpolation."""
    interpolator: ScalarPredictiveHermite


class FeatureFloatInterpolator(FeatureInterpolatorBase[float]):
    """Interpolator for scalar float values.

    Features:
    - Smooth interpolation of single float values
    - Velocity-based smoothing for natural motion using Hermite interpolation
    - Handles NaN values gracefully (propagates through interpolation)

    Suitable for:
    - motion_time (time since last significant motion)
    - confidence scores
    - temporal metrics
    - any other single float values

    Note:
        Uses ScalarPredictiveHermite internally for cubic Hermite interpolation.
    """

    def _create_state(self) -> FloatFilterState:
        """Create initial filter state for interpolation."""
        return FloatFilterState(
            interpolator=ScalarPredictiveHermite(
                input_rate=self._input_rate,
                alpha_v=self._alpha_v
            )
        )

    def _add_sample(self, feature: float) -> None:
        """Add float sample to interpolator."""
        state: FloatFilterState = self._state
        state.interpolator.add_sample(feature)

    def _interpolate(self, current_time: float | None) -> float:
        """Generate interpolated float at current time."""
        state: FloatFilterState = self._state

        # Update interpolator to current time
        state.interpolator.update(current_time)

        # Return interpolated value
        return state.interpolator.interpolated_value

    def _on_alpha_v_changed(self, value: float) -> None:
        """Update interpolator when alpha_v changes."""
        self._state.interpolator.alpha_v = value