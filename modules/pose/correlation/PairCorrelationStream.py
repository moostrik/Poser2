import math
import numpy as np
from functools import cached_property
from dataclasses import dataclass

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
    pair_history: dict[tuple[int, int], np.ndarray]
    capacity: int

    @property
    def is_empty(self) -> bool:
        """Check if the snapshot has no data."""
        return len(self.pair_history) == 0

    @cached_property
    def _sorted_pairs_by_similarity(self) -> list[tuple[tuple[int, int], float]]:
        """Pre-compute sorted pairs by average similarity (descending)."""
        result: list[tuple[tuple[int, int], float]] = []

        for pair_id, similarities in self.pair_history.items():
            avg_similarity = float(np.nanmean(similarities))
            if not np.isnan(avg_similarity):
                result.append((pair_id, avg_similarity))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_top_pairs(self, n: int = 5) -> list[tuple[int, int]]:
        """Get the top N most similar pose pairs, sorted by pair IDs."""
        top_n: list[tuple[tuple[int, int], float]] = self._sorted_pairs_by_similarity[:n]
        return sorted([pair_id for pair_id, _ in top_n])

    def get_similarities(self, pair_id: tuple[int, int], max_samples: int | None = None) -> np.ndarray:
        """Get similarity array for a pair, optionally truncated to max_samples."""
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")

        similarities: np.ndarray = self.pair_history[pair_id]

        if max_samples is None or len(similarities) <= max_samples:
            return similarities

        return similarities[-max_samples:]


class PairCorrelationStream:
    """Non-threaded correlation stream processor with circular buffers."""

    def __init__(self, capacity: int) -> None:
        """Initialize stream with fixed buffer capacity."""
        self.buffer_capacity: int = capacity
        self._pair_history: dict[tuple[int, int], tuple[np.ndarray, int]] = {}
        self._output_data: PairCorrelationStreamData | None = None
        self._snapshot_dirty: bool = True
        self.hot_reload = HotReloadMethods(self.__class__, True, True)

    def update(self, batch: PairCorrelationBatch,
               metric: SimilarityMetric = SimilarityMetric.GEOMETRIC_MEAN) -> None:
        """Update stream with new correlation batch.

        Args:
            batch: Batch of pair correlations to process
            metric: Similarity metric to use (default: geometric mean)
        """
        current_pairs: set[tuple[int, int]] = set()

        # Process new correlations
        for pair_corr in batch.pair_correlations:
            pair_id: tuple[int, int] = (min(pair_corr.pair_id), max(pair_corr.pair_id))
            current_pairs.add(pair_id)

            similarity: float = pair_corr.get_metric_value(metric)

            if pair_id in self._pair_history:
                similarities, write_idx = self._pair_history[pair_id]
                similarities[write_idx] = similarity
                write_idx = (write_idx + 1) % self.buffer_capacity
                self._pair_history[pair_id] = (similarities, write_idx)
            else:
                similarities = np.full(self.buffer_capacity, np.nan, dtype=np.float32)
                similarities[0] = similarity
                self._pair_history[pair_id] = (similarities, 1)

        # Remove pairs not in current batch
        for pair_id in list(self._pair_history.keys()):
            if pair_id not in current_pairs:
                del self._pair_history[pair_id]

        self._snapshot_dirty = True

    def get_stream_data(self) -> PairCorrelationStreamData:
        """Get lazily computed immutable snapshot of current stream state."""
        if self._snapshot_dirty or self._output_data is None:
            snapshot_history: dict[tuple[int, int], np.ndarray] = {}

            for pair_id, (similarities, write_idx) in self._pair_history.items():
                # Always concatenate full buffer - NaN values at end if not filled yet
                snapshot_history[pair_id] = np.concatenate([
                    similarities[write_idx:],
                    similarities[:write_idx]
                ])

            for arr in snapshot_history.values():
                arr.flags.writeable = False

            self._output_data = PairCorrelationStreamData(
                pair_history=snapshot_history,
                capacity=self.buffer_capacity
            )
            self._snapshot_dirty = False

        return self._output_data
