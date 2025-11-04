# Standard library imports
from dataclasses import dataclass

# Third-party imports
import numpy as np

# Pose imports
from .FeatureInterpolatorBase import FeatureInterpolatorBase
from ...features.PoseAngleSymmetry import PoseAngleSymmetryData, SYMM_NUM_JOINTS

# Local application imports
from modules.utils.Interpolation import VectorPredictiveHermite


@dataclass
class SymmetryFilterState:
    """State for angle symmetry interpolation."""
    # Interpolator for symmetry scores [0, 1]
    interpolator: VectorPredictiveHermite

    # Store last valid scores for reconstruction
    last_scores: np.ndarray  # shape: (SYMM_NUM_JOINTS,)


class FeatureSymmetryInterpolator(FeatureInterpolatorBase[PoseAngleSymmetryData]):
    """Interpolator for joint angle symmetry scores.

    Features:
    - Linear interpolation for symmetry scores [0, 1]
    - Velocity-based smoothing for natural motion using Hermite interpolation
    - Handles missing/invalid symmetry values gracefully (NaN propagation)
    - Supports score-based validity tracking

    Note:
        Uses VectorPredictiveHermite internally (no angular wrapping needed since
        symmetry values are already in [0, 1] range).
    """

    def _create_state(self) -> SymmetryFilterState:
        """Create initial filter state for interpolation."""
        return SymmetryFilterState(
            interpolator=VectorPredictiveHermite(
                input_rate=self._input_rate,
                vector_size=SYMM_NUM_JOINTS,
                alpha_v=self._alpha_v
            ),
            last_scores=np.zeros(SYMM_NUM_JOINTS, dtype=np.float32)
        )

    def _add_sample(self, feature: PoseAngleSymmetryData) -> None:
        """Add symmetry sample to interpolator."""
        state: SymmetryFilterState = self._state

        # Add samples to interpolator
        # Note: VectorPredictiveHermite handles NaN values automatically
        state.interpolator.add_sample(feature.values)

        # Store scores for reconstruction
        state.last_scores = feature.scores

    def _interpolate(self, current_time: float | None) -> PoseAngleSymmetryData:
        """Generate interpolated symmetry scores at current time."""
        state: SymmetryFilterState = self._state

        # Update interpolator to current time
        state.interpolator.update(current_time)

        # Get interpolated symmetry values
        interpolated_values: np.ndarray = state.interpolator.interpolated_value

        # Clamp to valid range [0, 1] (in case of extrapolation overshoot)
        interpolated_values = np.clip(interpolated_values, 0.0, 1.0)

        # Derive scores from validity (NaN -> 0.0 score)
        # This ensures data integrity constraint: NaN values must have 0.0 scores
        has_nan: np.ndarray = np.isnan(interpolated_values)
        interpolated_scores: np.ndarray = np.where(
            ~has_nan,
            state.last_scores,  # Preserve original scores for valid symmetry values
            0.0
        ).astype(np.float32)

        return PoseAngleSymmetryData(values=interpolated_values, scores=interpolated_scores)

    def _on_alpha_v_changed(self, value: float) -> None:
        """Update interpolator when alpha_v changes."""
        self._state.interpolator.alpha_v = value