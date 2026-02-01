"""Temporal pose-tolerant motion synchrony.

Temporal pose-tolerant motion synchrony is defined as the maximum whole-body
pose similarity observed within a short temporal window.

The algorithm compares all time slices from window A against all time slices
from window B, tracking the best similarity found for each joint across all
T_a × T_b combinations. This allows for temporal offset tolerance in pose
matching.
"""

import numpy as np

from modules.pose.nodes.windows.WindowNode import FeatureWindow


def compute_temporal_synchrony(
    window_a: FeatureWindow,
    window_b: FeatureWindow,
    exponent: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute temporal pose-tolerant motion synchrony between two windows.

    Compares all T_a × T_b time slice combinations, tracking the best similarity
    per joint. Fully vectorized for performance.

    Algorithm:
        1. Compute angular differences for all (T_a, T_b) pairs at once
        2. Convert to similarity scores: (1 - |diff| / π) ^ exponent
        3. Take max similarity per joint across all time combinations
        4. Compute confidence as min of mask values for best matches

    Args:
        window_a: First feature window with shape (T_a, feature_len)
        window_b: Second feature window with shape (T_b, feature_len)
        exponent: Decay rate for similarity (1.0 = linear, 2.0 = quadratic)

    Returns:
        Tuple of (values, scores) arrays, each of shape (feature_len,):
        - values: Best similarity found for each joint [0, 1]
        - scores: Confidence scores (min confidence from best matches)
    """
    # Get values: (T_a, feature_len) and (T_b, feature_len)
    values_a = window_a.values  # (T_a, F)
    values_b = window_b.values  # (T_b, F)
    mask_a = window_a.mask      # (T_a, F)
    mask_b = window_b.mask      # (T_b, F)

    # Broadcast to compute all pairwise angular differences
    # values_a[:, None, :] -> (T_a, 1, F)
    # values_b[None, :, :] -> (1, T_b, F)
    # Result: (T_a, T_b, F)
    raw_diff = values_a[:, None, :] - values_b[None, :, :]

    # Wrap angular difference to [-π, π]
    angular_diff = np.mod(raw_diff + np.pi, 2 * np.pi) - np.pi

    # Compute similarity: (1 - |diff| / π) ^ exponent
    # Shape: (T_a, T_b, F)
    similarity = np.power(1.0 - np.abs(angular_diff) / np.pi, exponent)

    # Compute confidence mask (both must be valid)
    # mask_a[:, None, :] -> (T_a, 1, F)
    # mask_b[None, :, :] -> (1, T_b, F)
    # Result: (T_a, T_b, F)
    valid_mask = mask_a[:, None, :] & mask_b[None, :, :]

    # Set invalid comparisons to -inf so they don't win max
    similarity = np.where(valid_mask, similarity, -np.inf)

    # Find best similarity per joint across all time combinations
    # Max over (T_a, T_b) axes -> shape (F,)
    best_similarity = np.max(similarity, axis=(0, 1))

    # Handle case where all comparisons were invalid
    best_similarity = np.where(np.isinf(best_similarity), 0.0, best_similarity)

    # For scores, use the proportion of valid comparisons per joint
    # This gives confidence based on how much data was available
    valid_count = np.sum(valid_mask, axis=(0, 1))  # (F,)
    total_count = valid_mask.shape[0] * valid_mask.shape[1]
    scores = (valid_count / total_count).astype(np.float32)

    return best_similarity.astype(np.float32), scores


def compute_harmonic_mean(similarities: np.ndarray, scores: np.ndarray, eps: float = 1e-6) -> float:
    """Compute confidence-weighted harmonic mean of similarities.

    Args:
        similarities: Per-joint similarity values (feature_len,)
        scores: Per-joint confidence scores (feature_len,)
        eps: Small value to prevent division by zero

    Returns:
        Harmonic mean of similarities, weighted by confidence.
        Returns 0.0 if no joints have sufficient confidence.
    """
    # Only include joints with non-zero confidence
    active_mask = scores > eps
    active_count = np.sum(active_mask)

    if active_count == 0:
        return 0.0

    # Compute harmonic mean: n / sum(1 / (s + eps))
    active_sims = similarities[active_mask]
    reciprocal_sum = np.sum(1.0 / (active_sims + eps))

    return float(active_count / reciprocal_sum)