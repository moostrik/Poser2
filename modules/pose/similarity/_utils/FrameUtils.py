import numpy as np

from modules.pose.features import Angles, AngleLandmark


class FrameUtils:
    """Utilities for comparing poses and computing similarity metrics."""

    @staticmethod
    def compute_similarity(angles1: Angles, angles2: Angles,
                          exponent: float = 2.0) -> Angles:
        """Compute angle-based similarity with confidence scores.

        Uses exponential decay based on angular difference:
            similarity = (1 - |angle_diff| / π) ^ exponent

        Args:
            angles1: First angle feature
            angles2: Second angle feature
            exponent: Decay rate (higher = stricter matching)
                     - 1.0 = linear decay
                     - 2.0 = quadratic decay (default)
                     - 3.0 = cubic decay (very strict)

        Returns:
            AngleFeature where:
            - values[i] = similarity score [0, 1] for joint i
              (1.0 = identical, 0.0 = maximally different)
            - scores[i] = confidence in this similarity measurement
              (minimum confidence from both source angles)

            NOTE: Returned .values are similarity percentages [0, 1],
                  NOT angles in radians!

        Examples:
            >>> sim = AngleSimilarity.compute_similarity(ref, current, exponent=2.0)
            >>>
            >>> # Check high-confidence match
            >>> if sim[AngleLandmark.left_elbow] > 0.8 and sim.get_score(AngleLandmark.left_elbow) > 0.7:
            ...     print("High-confidence good match!")
            >>>
            >>> # Get all similarities and scores
            >>> similarities = sim.values  # [0, 1] range
            >>> confidences = sim.scores   # Reliability of each comparison
        """
        # Compute angular differences with confidence scores
        delta = angles1.subtract(angles2)

        # Transform angle differences to (absolute) similarity scores
        similarity_values = np.power(1.0 - np.abs(delta.values) / np.pi, exponent)

        # Return as AngleFeature to preserve confidence scores
        return Angles(values=similarity_values, scores=delta.scores)

    @staticmethod
    def weighted_similarity(angles1: Angles, angles2: Angles,
                           weights: dict[AngleLandmark, float],
                           exponent: float = 2.0,
                           min_confidence: float = 0.0) -> float:
        """Compute weighted overall similarity score.

        Args:
            angles1: First angle feature
            angles2: Second angle feature
            weights: Importance weight for each joint (0-1)
            exponent: Decay rate for similarity calculation (default: 2.0)
            min_confidence: Minimum confidence to include joint (default: 0.0)
                           Set higher (e.g., 0.5) to ignore uncertain joints

        Returns:
            Overall weighted similarity score [0, 1]
            Returns 0.0 if no joints meet confidence threshold

        Examples:
            >>> weights = {
            ...     AngleLandmark.left_elbow: 1.0,      # Most important
            ...     AngleLandmark.left_shoulder: 0.8,
            ...     AngleLandmark.left_knee: 0.9,
            ...     AngleLandmark.head: 0.5,            # Less important
            ... }
            >>>
            >>> # Basic weighted similarity
            >>> score = AngleSimilarity.weighted_similarity(ref, current, weights)
            >>>
            >>> # Only trust high-confidence comparisons
            >>> score = AngleSimilarity.weighted_similarity(
            ...     ref, current, weights,
            ...     exponent=2.0, min_confidence=0.7
            ... )
        """
        # Compute similarity with confidence scores
        sim = FrameUtils.compute_similarity(angles1, angles2, exponent)

        # Apply weights and confidence threshold
        weighted_sum = 0.0
        weight_sum = 0.0

        for joint, weight in weights.items():
            similarity_value = sim[joint]
            confidence = sim.get_score(joint)

            # Include if valid, above confidence threshold
            if not np.isnan(similarity_value) and confidence >= min_confidence:
                weighted_sum += similarity_value * weight * confidence  # ✅ Weight by confidence too
                weight_sum += weight * confidence

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0