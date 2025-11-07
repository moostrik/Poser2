# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.filter.interpolation._FeatureInterpolatorBase import FeatureInterpolatorBase
from modules.pose.features.PoseAngles import PoseAngleData, ANGLE_NUM_JOINTS
from modules.pose.filter.interpolation.predictive.VectorPredictiveChaseInterpolator import AnglePredictiveChaseInterpolator

from modules.pose.filter.interpolation.PoseInterpolatorConfig import PoseInterpolatorConfig



@dataclass
class AngleFilterState:
    """Per-joint state for angle interpolation."""
    # Interpolator for angular values with proper wrapping
    interpolator: AnglePredictiveChaseInterpolator

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
        Uses VectorAngle internally for smooth interpolation with
        predictive extrapolation to minimize latency.
    """

    def _create_state(self) -> AngleFilterState:
        """Create initial filter state for interpolation."""
        return AngleFilterState(
            interpolator=AnglePredictiveChaseInterpolator(
                input_frequency=self._config.frequency,
                vector_size=ANGLE_NUM_JOINTS,
                responsiveness=self._config.responsiveness,
                friction=self._config.friction
            ),
            last_scores=np.zeros(ANGLE_NUM_JOINTS, dtype=np.float32)
        )

    def _add_sample(self, feature: PoseAngleData) -> None:
        """Add angle sample to interpolator."""
        state: AngleFilterState = self._state

        # Add samples to interpolator
        # Note: VectorAngle handles NaN values and angular wrapping automatically
        state.interpolator.add_sample(feature.values)

        # Store scores for reconstruction
        state.last_scores = feature.scores

    def _interpolate(self, current_time: float | None) -> PoseAngleData:
        """Generate interpolated angles at current time."""
        state: AngleFilterState = self._state

        # Update interpolator to current time
        state.interpolator.update(current_time)

        # Get interpolated angles
        interpolated_values: np.ndarray = state.interpolator.get_interpolated()

        # Derive scores from validity (NaN -> 0.0 score)
        # This ensures data integrity constraint: NaN values must have 0.0 scores
        has_nan: np.ndarray = np.isnan(interpolated_values)
        interpolated_scores: np.ndarray = np.where(
            ~has_nan,
            state.last_scores,  # Preserve original scores for valid angles
            0.0
        ).astype(np.float32)

        return PoseAngleData(values=interpolated_values, scores=interpolated_scores)

    def _on_config_changed(self) -> None:
        """Update interpolator when config changes."""
        self._state.interpolator.input_frequency = self._config.frequency
        self._state.interpolator.responsiveness = self._config.responsiveness
        self._state.interpolator.friction = self._config.friction