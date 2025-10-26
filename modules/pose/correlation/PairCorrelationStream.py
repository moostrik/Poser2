import math
import numpy as np
from functools import cached_property
from dataclasses import dataclass
from typing import Dict, Tuple, Set

from modules.pose.correlation.PairCorrelation import PairCorrelationBatch, SimilarityMetric
from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass(frozen=True)
class PairCorrelationStreamData:
    """Immutable snapshot of pose pair correlation history.

    Stores time-series similarity data for each pose pair.
    NaN values indicate frames where the pair couldn't be compared.

    Attributes:
        pair_history: Dictionary mapping pair IDs to similarity arrays
        capacity: Maximum number of samples stored per pair
    """
    pair_history: Dict[Tuple[int, int], np.ndarray]  # pair_id -> similarities array
    capacity: int

    @property
    def is_empty(self) -> bool:
        """Check if the snapshot has no data."""
        return len(self.pair_history) == 0

    @cached_property
    def _sorted_pairs_by_similarity(self) -> list[Tuple[Tuple[int, int], float]]:
        """Pre-compute sorted pairs by average similarity (descending)."""
        result: list[Tuple[Tuple[int, int], float]] = []

        for pair_id, similarities in self.pair_history.items():
            avg_similarity = np.nanmean(similarities)
            if not np.isnan(avg_similarity):
                result.append((pair_id, float(avg_similarity)))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_top_pairs(self, n: int = 5) -> list[Tuple[int, int]]:
        """Get the top N most similar pose pairs, sorted by pair IDs."""
        top_n = self._sorted_pairs_by_similarity[:n]
        return sorted([pair_id for pair_id, _ in top_n])

    def get_metric_window_array(self, pair_id: Tuple[int, int],
                                 max_samples: int | None = None) -> np.ndarray | None:
        """Get similarity array for a pair, optionally truncated to max_samples."""
        if pair_id not in self.pair_history:
            return None

        similarities = self.pair_history[pair_id]

        if len(similarities) == 0:
            return None

        if max_samples is None or len(similarities) <= max_samples:
            return similarities

        return similarities[-max_samples:]

    @cached_property
    def _last_similarities(self) -> Dict[Tuple[int, int], float]:
        """Pre-compute last similarity values for all pairs."""
        result = {}
        for pair_id, similarities in self.pair_history.items():
            if len(similarities) > 0:
                last_val = similarities[-1]
                result[pair_id] = float(last_val) if not np.isnan(last_val) else math.nan
            else:
                result[pair_id] = math.nan
        return result

    def get_last_value_for(self, x: int, y: int) -> float:
        """Get the last similarity value for a pose pair."""
        pair_id = (min(x, y), max(x, y))
        return self._last_similarities.get(pair_id, math.nan)

    @cached_property
    def _precomputed_windows(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Pre-compute all similarity windows once per snapshot."""
        return {
            pair_id: similarities
            for pair_id, similarities in self.pair_history.items()
            if len(similarities) > 0
        }

    def get_top_pairs_with_windows(self, n: int, max_samples: int) -> list[Tuple[Tuple[int, int], np.ndarray]]:
        """Combined method: get top pairs with their similarity arrays."""
        top_pairs = self.get_top_pairs(n)
        result: list[Tuple[Tuple[int, int], np.ndarray]] = []

        for pair_id in top_pairs:
            if pair_id in self._precomputed_windows:
                values = self._precomputed_windows[pair_id]
                if len(values) > max_samples:
                    values = values[-max_samples:]
                result.append((pair_id, values))

        return result


class PairCorrelationStream:
    """Simple non-threaded correlation stream processor with circular buffers."""

    def __init__(self, capacity: int) -> None:
        self.buffer_capacity: int = capacity

        # Pre-allocated circular buffers: (similarities, write_index, filled)
        self._pair_history: Dict[Tuple[int, int], Tuple[np.ndarray, int, bool]] = {}

        # Initialize with empty snapshot
        self._output_data: PairCorrelationStreamData = PairCorrelationStreamData(
            pair_history={},
            capacity=capacity
        )

        self.hot_reload = HotReloadMethods(self.__class__, True, True)

    def update(self, batch: PairCorrelationBatch, metric: SimilarityMetric = SimilarityMetric.GEOMETRIC_MEAN) -> None:
        """Update stream with new correlation batch.

        Args:
            batch: Batch of pair correlations to process
            metric: Similarity metric to use (default: geometric mean)
        """
        # if batch.is_empty:
        #     return

        # Collect pair IDs from current batch
        current_pairs: Set[Tuple[int, int]] = set()

        # Process new correlations
        for pair_corr in batch.pair_correlations:
            # Inline canonical pair ID
            id1, id2 = pair_corr.pair_id
            pair_id = (min(id1, id2), max(id1, id2))
            current_pairs.add(pair_id)

            similarity = pair_corr.get_metric_value(metric)

            if pair_id in self._pair_history:
                similarities, write_idx, filled = self._pair_history[pair_id]

                # Write to circular buffer
                similarities[write_idx] = similarity

                # Update write index (circular)
                write_idx = (write_idx + 1) % self.buffer_capacity

                # Mark as filled once we've wrapped around
                if write_idx == 0:
                    filled = True

                self._pair_history[pair_id] = (similarities, write_idx, filled)
            else:
                # Create new pre-allocated buffer
                similarities: np.ndarray = np.full(self.buffer_capacity, np.nan, dtype=np.float32)
                similarities[0] = similarity

                self._pair_history[pair_id] = (similarities, 1, False)

        # Remove pairs not in current batch
        for pair_id in list(self._pair_history.keys()):
            if pair_id not in current_pairs:
                del self._pair_history[pair_id]

        # Create output snapshot
        snapshot_history: Dict[Tuple[int, int], np.ndarray] = {}

        for pair_id, (similarities, write_idx, filled) in self._pair_history.items():
            sim_copy = np.concatenate([similarities[write_idx:], similarities[:write_idx]])

            snapshot_history[pair_id] = sim_copy

        self._output_data = PairCorrelationStreamData(
            pair_history=snapshot_history,
            capacity=self.buffer_capacity
        )

    def get_stream_data(self) -> PairCorrelationStreamData:
        """Get the latest processed stream data. Always returns a valid snapshot."""
        return self._output_data
