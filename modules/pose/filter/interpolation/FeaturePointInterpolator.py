# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.filters.interpolation.FeatureInterpolatorBase import FeatureInterpolatorBase
from modules.pose.features.PosePoints import PosePointData, POSE_NUM_JOINTS

# Local application imports
from modules.utils.Interpolation import VectorPredictiveHermite


@dataclass
class PointFilterState:
    """Per-joint state for point interpolation."""
    # Separate interpolators for x and y coordinates
    interpolator_x: VectorPredictiveHermite
    interpolator_y: VectorPredictiveHermite

    # Store last valid scores for reconstruction
    last_scores: np.ndarray  # shape: (POSE_NUM_JOINTS,)


class FeaturePointInterpolator(FeatureInterpolatorBase[PosePointData]):
    """Interpolator for 2D pose keypoint data with per-joint filtering.

    Features:
    - Independent filtering per joint coordinate (x, y)
    - Velocity-based smoothing for natural motion using Hermite interpolation
    - Handles missing/invalid joints gracefully (NaN propagation)
    - Supports score-based validity tracking

    Note:
        Uses VectorPredictiveHermite internally for smooth interpolation with
        predictive extrapolation to minimize latency.
    """

    def _create_state(self) -> PointFilterState:
        """Create initial filter state for interpolation."""
        return PointFilterState(
            interpolator_x=VectorPredictiveHermite(
                input_rate=self._input_rate,
                vector_size=POSE_NUM_JOINTS,
                alpha_v=self._alpha_v
            ),
            interpolator_y=VectorPredictiveHermite(
                input_rate=self._input_rate,
                vector_size=POSE_NUM_JOINTS,
                alpha_v=self._alpha_v
            ),
            last_scores=np.zeros(POSE_NUM_JOINTS, dtype=np.float32)
        )

    def _add_sample(self, feature: PosePointData) -> None:
        """Add keypoint sample to interpolators."""
        state: PointFilterState = self._state

        # Extract x and y coordinates
        x_values: np.ndarray = feature.values[:, 0]
        y_values: np.ndarray = feature.values[:, 1]

        # Add samples to respective interpolators
        # Note: VectorPredictiveHermite handles NaN values automatically
        state.interpolator_x.add_sample(x_values)
        state.interpolator_y.add_sample(y_values)

        # Store scores for reconstruction
        state.last_scores = feature.scores

    def _interpolate(self, current_time: float | None) -> PosePointData:
        """Generate interpolated keypoints at current time."""
        state: PointFilterState = self._state

        # Update interpolators to current time
        state.interpolator_x.update(current_time)
        state.interpolator_y.update(current_time)

        # Get interpolated coordinates
        interpolated_x: np.ndarray = state.interpolator_x.interpolated_value
        interpolated_y: np.ndarray = state.interpolator_y.interpolated_value

        # Reconstruct values array (POSE_NUM_JOINTS, 2)
        interpolated_values: np.ndarray = np.stack([interpolated_x, interpolated_y], axis=1)

        # Derive scores from validity (NaN -> 0.0 score)
        # This ensures data integrity constraint: NaN values must have 0.0 scores
        has_nan: np.ndarray = np.isnan(interpolated_values).any(axis=1)
        interpolated_scores: np.ndarray = np.where(
            ~has_nan,
            state.last_scores,  # Preserve original scores for valid joints
            0.0
        ).astype(np.float32)

        return PosePointData(values=interpolated_values, scores=interpolated_scores)

    def _on_alpha_v_changed(self, value: float) -> None:
        self._state.interpolator_x.alpha_v = value
        self._state.interpolator_y.alpha_v = value