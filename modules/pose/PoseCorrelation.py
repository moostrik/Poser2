import numpy as np
import pandas as pd
from collections import deque
from typing import Callable, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field

class PosePairCorrelation:
    def __init__(self, id_1: int, id_2: int, joint_correlations: dict[str, float]) -> None:
        self.id_1 = id_1
        self.id_2 = id_2
        self.joint_correlations: dict[str, float] = joint_correlations
        self.similarity_score: float = float(np.mean(list(joint_correlations.values()))) if joint_correlations else 0.0

class PoseCorrelationBatch:
    def __init__(self) -> None:
        """Collection of DTW results from a single analysis run."""
        self._pair_correlations: list[PosePairCorrelation] = []
        self._timestamp: pd.Timestamp = pd.Timestamp.now()
        self._similarity: float = 0.0

    @property
    def is_empty(self) -> bool:
        """Check if the batch has no results."""
        return len(self._pair_correlations) == 0

    @property
    def count(self) -> int:
        """Return the number of windows in the batch."""
        return len(self._pair_correlations)

    @property
    def timestamp(self) -> pd.Timestamp:
        return self._timestamp

    @property
    def similarity(self) -> float:
        return self._similarity

    def add_result(self, result: PosePairCorrelation) -> None:
        """Add a PoseCorrelation result to the batch."""
        self._pair_correlations.append(result)
        self._similarity = sum(r.similarity_score for r in self._pair_correlations) / len(self._pair_correlations)

        # Sort by similarity score
        self._pair_correlations.sort(key=lambda r: r.similarity_score, reverse=True)

    def get_most_similar_pair(self) -> Optional[PosePairCorrelation]:
        """Return the pair with highest similarity score."""
        if not self._pair_correlations:
            return None
        return max(self._pair_correlations, key=lambda r: r.similarity_score)

PairCorrelationBatchCallback = Callable[[PoseCorrelationBatch], None]


@dataclass
class PairData:
    """Class for storing all data related to a pose pair."""
    history: deque = field(default_factory=lambda: deque(maxlen=250))
    activity_count: int = 0
    last_active_time: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    @property
    def avg_similarity(self) -> float:
        """Calculate average similarity over history."""
        if not self.history:
            return 0.0
        return float(np.mean([sim for _, sim in self.history]))

    @property
    def consistency(self) -> float:
        """Calculate consistency (inverse of standard deviation)."""
        if len(self.history) < 2:
            return 0.0
        similarities: list[float] = [sim for _, sim in self.history]
        return float(1.0 / (np.std(similarities) + 0.001))

class PoseCorrelationHistory:
    def __init__(self, max_history: int = 250, inactive_timeout_seconds: float = 1.0):
        """
        Tracks pose correlation batches over time in a rolling window.

        Args:
            max_history: Maximum number of batches to store
            inactive_timeout_seconds: Remove pairs inactive for this many seconds
        """
        self.max_history: int = max_history
        self.inactive_timeout = pd.Timedelta(seconds=inactive_timeout_seconds)

        self.batches: deque[PoseCorrelationBatch] = deque(maxlen=max_history)

        # Single dictionary to store all pair data
        self.pairs: dict[str, PairData] = {}  # Maps pair_id -> PairData

    def add_batch(self, batch: PoseCorrelationBatch) -> None:
        """Add a new correlation batch to the history."""
        self.batches.append(batch)

        # Track which pairs are active in this batch
        active_pairs_in_batch: set[str] = set()

        if not batch.is_empty:
            # Update pair histories for pairs in this batch
            for corr in batch._pair_correlations:
                pair_id: str = self._get_pair_id(corr.id_1, corr.id_2)
                active_pairs_in_batch.add(pair_id)

                # Create entry if this is a new pair
                if pair_id not in self.pairs:
                    self.pairs[pair_id] = PairData(
                        history=deque(maxlen=self.max_history),
                        activity_count=0,
                        last_active_time=batch.timestamp
                    )

                # Update pair data
                self.pairs[pair_id].history.append((batch.timestamp, corr.similarity_score))
                self.pairs[pair_id].activity_count += 1
                self.pairs[pair_id].last_active_time = batch.timestamp

        # Update inactivity for pairs not in this batch
        inactive_pairs: Set[str] = set(self.pairs.keys()) - active_pairs_in_batch

        # For inactive pairs, add a zero-similarity entry to maintain continuity
        for pair_id in inactive_pairs:
            # Add a placeholder with zero similarity to show inactivity
            self.pairs[pair_id].history.append((batch.timestamp, 0.0))
            # Note: We don't increment activity count for inactive pairs

        # Prune inactive pairs
        self._prune_inactive_pairs(batch.timestamp)

    def _prune_inactive_pairs(self, current_time: pd.Timestamp) -> None:
        """
        Remove pairs that have been inactive for too long.

        Args:
            current_time: The timestamp of the current batch
        """
        pairs_to_remove = []

        for pair_id, pair_data in self.pairs.items():
            # Calculate how long the pair has been inactive
            inactive_duration = current_time - pair_data.last_active_time

            # If the pair has been inactive for longer than the timeout
            if inactive_duration > self.inactive_timeout:
                pairs_to_remove.append(pair_id)

        # Remove inactive pairs
        for pair_id in pairs_to_remove:
            del self.pairs[pair_id]

    def get_top_pairs(self, n: int = 5) -> Dict[str, float]:
        """Get the top N pairs by average similarity over the entire history."""
        if not self.pairs:
            return {}

        # Calculate average similarity for each pair
        avg_similarities = {pair_id: pair_data.avg_similarity
                           for pair_id, pair_data in self.pairs.items()}

        # Sort by average similarity and return top N
        sorted_pairs = sorted(avg_similarities.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_pairs[:n])

    def get_most_consistent_pairs(self, n: int = 5) -> Dict[str, float]:
        """Get the N pairs with the most consistent similarity (lowest standard deviation)."""
        if not self.pairs:
            return {}

        # Calculate consistency for each pair
        consistency = {pair_id: pair_data.consistency
                      for pair_id, pair_data in self.pairs.items()}

        # Sort by consistency and return top N
        sorted_pairs = sorted(consistency.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_pairs[:n])

    def get_pair_data_for_visualization(self) -> Dict[str, Dict]:
        """
        Returns all pair data organized for visualization.

        Returns:
            Dict with timestamps as keys, containing all pair similarities for that time
        """
        result = {}

        # Collect all timestamps
        all_timestamps = set()
        for pair_data in self.pairs.values():
            all_timestamps.update(ts for ts, _ in pair_data.history)

        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)

        # Create a dict with timestamps as keys
        for ts in sorted_timestamps:
            result[ts] = {"overall": 0.0, "pairs": {}}

        # Fill in pair data
        for pair_id, pair_data in self.pairs.items():
            for ts, sim in pair_data.history:
                if ts in result:
                    result[ts]["pairs"][pair_id] = sim

        # Calculate overall similarity for each timestamp
        for ts in result:
            if result[ts]["pairs"]:
                result[ts]["overall"] = np.mean(list(result[ts]["pairs"].values()))

        return result

    @staticmethod
    def _get_pair_id(id1: int, id2: int) -> str:
        """Create a consistent string ID for a pair of IDs."""
        return f"{min(id1, id2)}-{max(id1, id2)}"

    @staticmethod
    def get_pair_ids_from_string(pair_id: str) -> Tuple[int, int]:
        """Convert a pair ID string back to individual IDs."""
        id1_str, id2_str = pair_id.split('-')
        return int(id1_str), int(id2_str)