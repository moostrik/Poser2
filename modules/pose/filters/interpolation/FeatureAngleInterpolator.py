# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.filters.interpolation.FeatureInterpolatorBase import FeatureInterpolatorBase
from modules.pose.features.PoseAngles import PoseAngleData, ANGLE_NUM_JOINTS

# Local application imports
from modules.utils.Interpolation import VectorPredictiveAngleHermite


@dataclass
class AngleFilterState:
    """Per-joint state for angle interpolation."""
    # Interpolator for angular values with proper wrapping
    interpolator: VectorPredictiveAngleHermite

    # Store last valid scores for reconstruction
    last_scores: np.ndarray  # shape: (ANGLE_NUM_JOINTS,)


class FeatureAngleInterpolator(FeatureInterpolatorBase[PoseAngleData]):
    """Interpolator for joint angle data with proper angular wrapping.

    Features:
    - Angular interpolation with proper wrapping around [-π, π]
    - Velocity-based smoothing for natural motion using Hermite interpolation
    - Handles missing/invalid angles gracefully (NaN propagation)
    - Supports score-based validity tracking

    Note:
        Uses VectorPredictiveAngleHermite internally for smooth interpolation with
        predictive extrapolation to minimize latency.
    """

    def _create_state(self) -> AngleFilterState:
        """Create initial filter state for interpolation."""
        return AngleFilterState(
            interpolator=VectorPredictiveAngleHermite(
                input_rate=self._input_rate,
                vector_size=ANGLE_NUM_JOINTS,
                alpha_v=self._alpha_v
            ),
            last_scores=np.zeros(ANGLE_NUM_JOINTS, dtype=np.float32)
        )

    def _add_sample(self, feature: PoseAngleData) -> None:
        """Add angle sample to interpolator."""
        state: AngleFilterState = self._state

        # Add samples to interpolator
        # Note: VectorPredictiveAngleHermite handles NaN values and angular wrapping automatically
        state.interpolator.add_sample(feature.values)

        # Store scores for reconstruction
        state.last_scores = feature.scores

    def _interpolate(self, current_time: float | None) -> PoseAngleData:
        """Generate interpolated angles at current time."""
        state: AngleFilterState = self._state

        # Update interpolator to current time
        state.interpolator.update(current_time)

        # Get interpolated angles
        interpolated_values: np.ndarray = state.interpolator.interpolated_value

        # Derive scores from validity (NaN -> 0.0 score)
        # This ensures data integrity constraint: NaN values must have 0.0 scores
        has_nan: np.ndarray = np.isnan(interpolated_values)
        interpolated_scores: np.ndarray = np.where(
            ~has_nan,
            state.last_scores,  # Preserve original scores for valid angles
            0.0
        ).astype(np.float32)

        return PoseAngleData(values=interpolated_values, scores=interpolated_scores)

    def _on_alpha_v_changed(self, value: float) -> None:
        """Update interpolator when alpha_v changes."""
        self._state.interpolator.alpha_v = value