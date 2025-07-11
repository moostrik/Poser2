import numpy as np
import pandas as pd
from dataclasses import dataclass
from threading import Lock
from typing import Callable, Optional, Dict, Tuple

from modules.correlation.PairCorrelation import PairCorrelationBatch
from modules.Settings import Settings

@dataclass(frozen=True)
class PairCorrelationStreamData:
    pair_history: Dict[Tuple[int, int], pd.DataFrame]
    timestamp: pd.Timestamp

    def get_top_pairs(self, n: int = 5, duration: Optional[float] = None, min_similarity: Optional[float] = None) -> list[Tuple[int, int]]:
        """
        Get the top N most similar pose pairs based on average similarity scores.

        Args:
            n: Number of top pairs to return
            time_window: If provided, only consider data from the last X seconds
            min_similarity: If provided, only include pairs with similarity >= this value

        Returns:
            List of tuples containing (id_1, id_2) sorted by similarity
        """
        result: list[Tuple[int, int, float]] = []

        for pair_id, df in self.pair_history.items():
            if duration is not None and not df.empty:
                stream_end: pd.Timestamp = df.index[-1]
                window_cutoff: pd.Timestamp = stream_end - pd.Timedelta(seconds=duration)
                filtered_df: pd.DataFrame = df[df.index >= window_cutoff]
                if filtered_df.empty:
                    continue
                avg_similarity: float = filtered_df['similarity'].mean()
            else:
                avg_similarity = df['similarity'].mean()

            if min_similarity is not None and avg_similarity < min_similarity:
                continue

            # Store as (id_1, id_2, avg_similarity)
            result.append((pair_id[0], pair_id[1], avg_similarity))

        # Sort by similarity score (descending) and take top N
        result.sort(key=lambda x: x[2], reverse=True)
        return [(id1, id2) for id1, id2, _ in result[:n]]

    def get_metric_window(self, pair_id: Tuple[int, int], metric_name: str = "similarity", duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get the time series data for a specific joint of a pose pair.

        Args:
            pair_id: Tuple containing (id_1, id_2)
            metric_name: The metric to retrieve (default is 'similarity')
            time_window: If provided, only return data from the last X seconds

        Returns:
            Numpy array of joint values over time, or None if the pair is not found
        """
        if pair_id not in self.pair_history:
            return None

        df: pd.DataFrame = self.pair_history[pair_id]

        if duration is not None:
            stream_end: pd.Timestamp = df.index[-1]
            window_cutoff: pd.Timestamp = stream_end - pd.Timedelta(seconds=duration)
            df = df[df.index >= window_cutoff]

        if df.empty or metric_name not in df.columns:
            return None

        return df[metric_name].to_numpy()

class PairCorrelationStreamProcessor:
    def __init__(self, settings: Settings, history_duration: float = 60.0) -> None:
        """
        Tracks pose correlation batches over time in a rolling window using pandas DataFrames.

        Args:
            history_duration: Duration in seconds to maintain history
        """
        self.history_duration = pd.Timedelta(seconds=history_duration)
        self._pair_history: Dict[Tuple[int, int], pd.DataFrame] = {}

        self._update_callbacks_lock = Lock()
        self._update_callbacks: set[Callable[[PairCorrelationStreamData], None]] = set()

    def start(self) -> None:
        """ Initialize the processor. Currently does nothing but can be extended in the future. """
        pass

    def stop(self) -> None:
        """ Clean up resources if needed. Currently does nothing but can be extended in the future. """
        pass

    def add_batch(self, batch: PairCorrelationBatch) -> None:
        """
        Add a new batch to the history by extracting individual pair correlations.

        Args:
            batch: The PoseCorrelationBatch to process
        """
        if not batch.is_empty:

            timestamp: pd.Timestamp = batch.timestamp
            current_time: pd.Timestamp = pd.Timestamp.now()
            cutoff_time: pd.Timestamp = current_time - self.history_duration

            # Process each pair correlation in the batch
            for pair_corr in batch.pair_correlations:
                pair_id: Tuple[int, int] = pair_corr.pair_id

                # Create a new row with the similarity score and joint correlations
                new_data: Dict[str, float] = {'similarity': pair_corr.similarity_score}
                new_data.update(pair_corr.joint_correlations)

                # Create a DataFrame with one row
                new_row = pd.DataFrame(new_data, index=[timestamp])

                # Add to existing DataFrame or create a new one
                if pair_id in self._pair_history:
                    self._pair_history[pair_id] = pd.concat([self._pair_history[pair_id], new_row])
                else:
                    self._pair_history[pair_id] = new_row

                # Ensure the DataFrame is sorted by timestamp
                self._pair_history[pair_id] = self._pair_history[pair_id].sort_index()

            # Prune old data
            self._prune_history(cutoff_time)

            self._notify_update_callbacks()

    def _prune_history(self, cutoff_time: pd.Timestamp) -> None:
        """
        Remove data points older than the cutoff time from all pairs.

        Args:
            cutoff_time: Timestamp before which data should be removed
        """
        pairs_to_remove: list[Tuple[int,int]] = []

        for pair_id, df in self._pair_history.items():
            # Keep only rows with timestamps >= cutoff_time
            pruned_df: pd.DataFrame = df[df.index >= cutoff_time]

            if pruned_df.empty:
                # All data is old, mark for removal
                pairs_to_remove.append(pair_id)
            else:
                # Update with pruned data
                self._pair_history[pair_id] = pruned_df

        # Remove pairs with no remaining data
        for pair_id in pairs_to_remove:
            del self._pair_history[pair_id]

    def get_stream_data(self) -> PairCorrelationStreamData:
        """Return a snapshot of the current pair correlation stream data."""
        # Use the latest timestamp among all pairs, or pd.Timestamp.now() if empty
        if self._pair_history:
            latest_ts: pd.Timestamp = max(df.index[-1] for df in self._pair_history.values() if not df.empty)
        else:
            latest_ts = pd.Timestamp.now()
        return PairCorrelationStreamData(
            pair_history={k: v.copy() for k, v in self._pair_history.items()},
            timestamp=latest_ts
        )

    @staticmethod
    def get_canonical_pair_id(pair_id: Tuple[int, int]) -> Tuple[int, int]:
        """
        Create a canonical pair ID by ensuring the smaller ID is always first.
        """
        return (pair_id[0], pair_id[1]) if pair_id[0] <= pair_id[1] else (pair_id[1], pair_id[0])

    def add_stream_callback(self, callback: Callable[[PairCorrelationStreamData], None]) -> None:
        """ Register a callback to receive the current PairCorrelationStreamData. """
        with self._update_callbacks_lock:
            self._update_callbacks.add(callback)

    def _notify_update_callbacks(self) -> None:
        """ Call all registered callbacks with the current PairCorrelationStreamData. """
        stream_data = self.get_stream_data()
        with self._update_callbacks_lock:
            for callback in self._update_callbacks:
                callback(stream_data)
